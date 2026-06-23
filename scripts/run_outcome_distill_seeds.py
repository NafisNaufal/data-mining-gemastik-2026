#!/usr/bin/env python
"""Multi-seed evaluation of cross-modal distillation on the treatment-outcome target.

Trains the deployable mobile student (baseline vs distilled from the LUPI teacher) over
several seeds and reports mean +- std test AUROC plus a *paired* significance test on the
per-seed gain. Run after training the outcome teacher (scripts/train_teacher.py
--target outcome_unfavorable), which writes the soft labels this consumes.

Run:  CUDA_VISIBLE_DEVICES=2 python scripts/run_outcome_distill_seeds.py --seeds 1426 7 99 13 2024 31 --epochs 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats

from pusketb.config import load_config
from pusketb.models.distill import feature_distill_loss, kd_loss
from pusketb.train import metrics
from pusketb.train.image_data import CXRDataset, rows_for_split
from scripts.train_student_e2e import MobileStudent, teacher_maps, evaluate_split


def train_once(cfg, target, rows, mode, seed, args, device, teacher):
    torch.manual_seed(seed); np.random.seed(seed)
    distill = mode == "distill"
    tl, tr, rep_dim = teacher if distill else (None, None, None)

    ds_train = CXRDataset(rows["train"], args.img_size, True, tl, tr)
    ds_calib = CXRDataset(rows["calib"], args.img_size, False)
    ds_test = CXRDataset(rows["test"], args.img_size, False)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.workers, pin_memory=True)
    train_loader, calib_loader, test_loader = dl(ds_train, True), dl(ds_calib, False), dl(ds_test, False)

    model = MobileStudent(args.backbone, rep_dim if distill else None, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos = float((rows["train"]["y"] == 0).sum() / max((rows["train"]["y"] == 1).sum(), 1))
    pos_weight = torch.tensor(pos, device=device)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["image"].to(device); y = batch["y"].to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_weight)
            if distill:
                loss = loss + args.w_kd * kd_loss(out["logit"], batch["t_logit"].to(device), T=args.temp)
                loss = loss + args.w_feat * feature_distill_loss(out["distill_rep"], batch["t_rep"].to(device))
            loss.backward(); opt.step()
        sched.step()
        m = evaluate_split(model, calib_loader, rows["calib"], device)
        if m["cond_auroc"] > best:
            best, best_state, bad = m["cond_auroc"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return evaluate_split(model, test_loader, rows["test"], device)["cond_auroc"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="outcome_unfavorable")
    ap.add_argument("--backbone", default="mobilenetv3_small_100")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1426, 7, 99, 13, 2024, 31, 256, 777])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--w-kd", type=float, default=1.5)
    ap.add_argument("--w-feat", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=3.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default="data/processed/outcome_distill_seeds.csv")
    ap.add_argument("--teacher-tag", default="",
                    help="suffix to select an alternate teacher dir, e.g. '_imgonly' for the "
                         "unimodal-teacher regularization control")
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = pd.read_parquet(Path(cfg.paths.processed_dir) / "manifest.parquet")
    rows = {s: rows_for_split(cfg, manifest, s, args.target) for s in ("train", "calib", "test")}
    teacher = teacher_maps(cfg, args.target, "train", tag=args.teacher_tag)
    print(f"target={args.target} backbone={args.backbone}  "
          f"n_train={len(rows['train'])} n_calib={len(rows['calib'])} n_test={len(rows['test'])}  "
          f"train pos={rows['train']['y'].mean():.3f}  teacher_rep_dim={teacher[2]}")
    print(f"recipe: w_kd={args.w_kd} w_feat={args.w_feat} T={args.temp}")

    recs = []
    for seed in args.seeds:
        b = train_once(cfg, args.target, rows, "baseline", seed, args, device, teacher)
        d = train_once(cfg, args.target, rows, "distill", seed, args, device, teacher)
        recs.append({"seed": seed, "baseline": b, "distill": d, "delta": d - b})
        print(f"  seed {seed:>5}: baseline={b:.4f}  distill={d:.4f}  delta={d-b:+.4f}", flush=True)

    df = pd.DataFrame(recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    bm, bs = df["baseline"].mean(), df["baseline"].std()
    dm, ds = df["distill"].mean(), df["distill"].std()
    delt = df["delta"].to_numpy()
    wil = stats.wilcoxon(delt) if len(delt) >= 6 and np.any(delt != 0) else None
    tt = stats.ttest_rel(df["distill"], df["baseline"])
    n_win = int((delt > 0).sum())
    print("\n=== OUTCOME DISTILLATION (multi-seed, paired) ===")
    print(f"baseline : {bm:.4f} ± {bs:.4f}")
    print(f"distill  : {dm:.4f} ± {ds:.4f}")
    print(f"delta    : {delt.mean():+.4f} ± {delt.std():.4f}  ({n_win}/{len(delt)} seeds improve)")
    print(f"paired t : t={tt.statistic:.3f}  p={tt.pvalue:.4f}")
    if wil is not None:
        print(f"Wilcoxon : W={wil.statistic:.1f}  p={wil.pvalue:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
