#!/usr/bin/env python
"""End-to-end mobile student (timm backbone) trained on raw CXRs, with distillation.

This is the deployable, tablet-grade model (C5) and the setting where cross-modal
distillation can actually help: the student backbone is *trainable*, so the multimodal
teacher's soft labels + image-pathway representation can shape the learned features.

--mode baseline : train the mobile net on hard labels only.
--mode distill  : add soft-label KD + feature alignment to the (cached) multimodal teacher.

Run:  python scripts/train_student_e2e.py --target rif_resistant --backbone mobilenetv3_small_100 --mode distill
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pusketb.config import load_config
from pusketb.models.distill import feature_distill_loss, kd_loss
from pusketb.train import metrics
from pusketb.train.image_data import CXRDataset, rows_for_split


class MobileStudent(nn.Module):
    def __init__(self, backbone: str, teacher_rep_dim: int | None, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                           num_classes=0, in_chans=3)
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Linear(d, 1)
        self.distill_proj = nn.Linear(d, teacher_rep_dim) if teacher_rep_dim else None

    def forward(self, x):
        f = self.backbone(x)
        out = {"logit": self.head(f).squeeze(-1)}
        if self.distill_proj is not None:
            out["distill_rep"] = self.distill_proj(f)
        return out


def teacher_maps(cfg, target, split, tag=""):
    f = Path(cfg.paths.processed_dir) / "teacher" / (target + tag) / f"teacher_{split}.npz"
    z = np.load(f, allow_pickle=True)
    urls = z["series_url"].astype(str)
    rep_key = "img_rep" if "img_rep" in z.files else "rep"
    tl = {u: float(l) for u, l in zip(urls, z["logit"])}
    tr = {u: r.astype(np.float32) for u, r in zip(urls, z[rep_key])}
    rep_dim = z[rep_key].shape[1]
    return tl, tr, rep_dim


@torch.no_grad()
def evaluate_split(model, loader, rows, device):
    model.eval()
    probs = np.zeros(len(rows), dtype=np.float32)
    for batch in loader:
        p = torch.sigmoid(model(batch["image"].to(device))["logit"]).cpu().numpy()
        probs[batch["idx"].numpy()] = p
    return metrics.evaluate(probs, rows["y"].to_numpy(), rows["condition_id"].to_numpy(),
                            rows["lineage_group"].to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--backbone", default="mobilenetv3_small_100")
    ap.add_argument("--mode", choices=["baseline", "distill"], default="distill")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--w-kd", type=float, default=1.0)
    ap.add_argument("--w-feat", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=3.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=1426)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distill = args.mode == "distill"
    target = args.target or cfg.labels.target

    manifest = pd.read_parquet(Path(cfg.paths.processed_dir) / "manifest.parquet")
    rows = {s: rows_for_split(cfg, manifest, s, target) for s in ("train", "calib", "test")}
    print(f"target={target} backbone={args.backbone} mode={args.mode}  "
          f"n_train={len(rows['train'])} n_calib={len(rows['calib'])} n_test={len(rows['test'])}  "
          f"train pos={rows['train']['y'].mean():.3f}")

    tl = tr = None
    rep_dim = None
    if distill:
        tl, tr, rep_dim = teacher_maps(cfg, target, "train")

    ds_train = CXRDataset(rows["train"], args.img_size, True, tl, tr)
    ds_calib = CXRDataset(rows["calib"], args.img_size, False)
    ds_test = CXRDataset(rows["test"], args.img_size, False)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.workers, pin_memory=True, drop_last=False)
    train_loader, calib_loader, test_loader = dl(ds_train, True), dl(ds_calib, False), dl(ds_test, False)

    model = MobileStudent(args.backbone, rep_dim if distill else None,
                          pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos = float((rows["train"]["y"] == 0).sum() / max((rows["train"]["y"] == 1).sum(), 1))
    pos_weight = torch.tensor(pos, device=device)

    best_auroc, best_state, bad = -1.0, None, 0
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
        if m["cond_auroc"] > best_auroc:
            best_auroc = m["cond_auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"ep {ep:3d}  calib cond_auroc={m['cond_auroc']:.4f}  (best {best_auroc:.4f})")
        if bad >= args.patience:
            print(f"early stop at ep {ep}"); break

    model.load_state_dict(best_state)
    m = evaluate_split(model, test_loader, rows["test"], device)
    print(f"\n=== E2E STUDENT {args.backbone} ({args.mode}) test ===\n{metrics.format_metrics(m)}")

    out_dir = Path(cfg.paths.processed_dir) / "student_e2e" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "backbone": args.backbone, "mode": args.mode},
               out_dir / f"{args.backbone}_{args.mode}.pt")
    print(f"saved {out_dir/f'{args.backbone}_{args.mode}.pt'}")


if __name__ == "__main__":
    main()
