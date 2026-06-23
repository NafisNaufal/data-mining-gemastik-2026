#!/usr/bin/env python
"""Train the X-ray-only student, with or without cross-modal distillation (C1).

--mode baseline : image-only, hard labels only (the ablation baseline).
--mode distill  : adds soft-label KD + feature/contrastive alignment to the teacher's
                  fused representation (requires scripts/train_teacher.py to have run).

Prints test metrics with the L2/L4 lineage breakdown (C2) so baseline vs distilled is
directly comparable.

Run:  python scripts/train_student.py --mode distill [--w-kd 1 --w-feat 1 --w-con 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pusketb.config import load_config
from pusketb.models.distill import student_loss
from pusketb.models.heads import StudentNet
from pusketb.train import datasets, metrics


def _load_teacher_outputs(cfg, split, target):
    f = Path(cfg.paths.processed_dir) / "teacher" / target / f"teacher_{split}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing — run scripts/train_teacher.py first")
    z = np.load(f)
    # Feature-distill toward the image-recoverable pathway rep, not the genomics-saturated fused rep.
    feat_key = "img_rep" if "img_rep" in z.files else "rep"
    return z["logit"].astype(np.float32), z[feat_key].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", default=None, help="label column (default cfg.labels.target)")
    ap.add_argument("--mode", choices=["baseline", "distill"], default="distill")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--rep-dim", type=int, default=256)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--w-ce", type=float, default=1.0)
    ap.add_argument("--w-kd", type=float, default=1.0)
    ap.add_argument("--w-feat", type=float, default=1.0)
    ap.add_argument("--w-con", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1426)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distill = args.mode == "distill"
    target = args.target or cfg.labels.target

    data = datasets.load_all(cfg, target=target)
    data.pop("_spec")
    tr, ca, te = data["train"], data["calib"], data["test"]

    teacher_rep_dim = None
    t_logit = t_rep = None
    if distill:
        t_logit, t_rep = _load_teacher_outputs(cfg, "train", target)
        teacher_rep_dim = t_rep.shape[1]
        t_logit = torch.from_numpy(t_logit).to(device)
        t_rep = torch.from_numpy(t_rep).to(device)

    model = StudentNet(tr.x_img.shape[1], hidden=args.hidden, rep_dim=args.rep_dim,
                       teacher_rep_dim=teacher_rep_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = float((tr.y == 0).sum() / max((tr.y == 1).sum(), 1))
    pos_weight = torch.tensor(pos, device=device)

    xi = torch.from_numpy(tr.x_img).to(device)
    y = torch.from_numpy(tr.y).to(device)
    n = len(y)
    best_auroc, best_state, bad = -1.0, None, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            out_s = model(xi[idx])
            out_t = ({"logit": t_logit[idx], "rep": t_rep[idx]} if distill else None)
            loss, _ = student_loss(out_s, out_t, y[idx], pos_weight,
                                   w_ce=args.w_ce, w_kd=args.w_kd if distill else 0,
                                   w_feat=args.w_feat if distill else 0,
                                   w_con=args.w_con if distill else 0, T=args.temp)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(ca.x_img).to(device))["logit"]).cpu().numpy()
        m = metrics.evaluate(prob, ca.y, ca.condition_id, ca.lineage)
        if m["cond_auroc"] > best_auroc:
            best_auroc = m["cond_auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"ep {ep:3d}  calib {metrics.format_metrics(m)}")
        if bad >= args.patience:
            print(f"early stop at ep {ep} (best calib cond_auroc={best_auroc:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(te.x_img).to(device))["logit"]).cpu().numpy()
    m = metrics.evaluate(prob, te.y, te.condition_id, te.lineage)
    print(f"\n=== STUDENT ({args.mode}) test ===\n{metrics.format_metrics(m)}")

    out_dir = Path(cfg.paths.processed_dir) / "student" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mode": args.mode,
                "img_dim": tr.x_img.shape[1], "hidden": args.hidden, "rep_dim": args.rep_dim},
               out_dir / f"student_{args.mode}.pt")
    np.save(out_dir / f"student_{args.mode}_test_prob.npy", prob)
    print(f"saved {out_dir/f'student_{args.mode}.pt'}")


if __name__ == "__main__":
    main()
