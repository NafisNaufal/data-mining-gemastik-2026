#!/usr/bin/env python
"""Train the multimodal teacher (RAD-DINO image embedding ⊕ genomics → rifampicin).

Saves the checkpoint + per-split teacher logits/reps (consumed by the distilled student).
Requires cached embeddings for each split (scripts/cache_embeddings.py --split ...).

Run:  python scripts/train_teacher.py [--epochs 60] [--lr 1e-3]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pusketb.config import load_config
from pusketb.models.heads import TeacherNet
from pusketb.train import datasets, metrics


def _tensors(d, device):
    return (torch.from_numpy(d.x_img).to(device),
            torch.from_numpy(d.x_gen).to(device),
            torch.from_numpy(d.y).to(device))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", default=None, help="label column (default cfg.labels.target)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--rep-dim", type=int, default=256)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="soften targets so the teacher transmits dark knowledge")
    ap.add_argument("--gen-dropout", type=float, default=0.5,
                    help="prob. of zeroing the genomics vector per sample (modality dropout)")
    ap.add_argument("--aux-img-weight", type=float, default=0.5,
                    help="weight on the auxiliary image-only head (forces image pathway)")
    ap.add_argument("--gen-zero", action="store_true",
                    help="ablate the privileged modality entirely (genomics/clinical always "
                         "zeroed) -> image-only 'unimodal teacher' control for distillation")
    ap.add_argument("--out-tag", default="",
                    help="suffix appended to the teacher output dir (e.g. '_imgonly' for the "
                         "control), so it does not overwrite the main teacher")
    ap.add_argument("--seed", type=int, default=1426)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target = args.target or cfg.labels.target
    data = datasets.load_all(cfg, target=target)
    spec = data.pop("_spec")
    tr, ca, te = data["train"], data["calib"], data["test"]
    print(f"target={target}  dims: img={tr.x_img.shape[1]} gen={tr.x_gen.shape[1]}  "
          f"n_train={len(tr.y)} (pos {tr.y.mean():.3f}) n_calib={len(ca.y)} n_test={len(te.y)}")

    model = TeacherNet(tr.x_img.shape[1], tr.x_gen.shape[1],
                       hidden=args.hidden, rep_dim=args.rep_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = float((tr.y == 0).sum() / max((tr.y == 1).sum(), 1))
    pos_weight = torch.tensor(pos, device=device)
    eps = args.label_smoothing

    def bce_soft(logit, target):
        # label smoothing toward 0.5, with positive-class reweighting
        soft = target * (1 - eps) + 0.5 * eps
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logit, soft, pos_weight=pos_weight)

    xi, xg, y = _tensors(tr, device)
    if args.gen_zero:  # image-only teacher control: never let the privileged block contribute
        xg = torch.zeros_like(xg)
    n = len(y)
    best_auroc, best_state, bad = -1.0, None, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            xg_b = xg[idx]
            if args.gen_dropout > 0:  # modality dropout: zero genomics for some samples
                mask = (torch.rand(xg_b.size(0), 1, device=device) >= args.gen_dropout).float()
                xg_b = xg_b * mask
            out = model(xi[idx], xg_b)
            loss = bce_soft(out["logit"], y[idx]) + args.aux_img_weight * bce_soft(out["img_logit"], y[idx])
            loss.backward()
            opt.step()

        # Calib selection on condition-level AUROC.
        model.eval()
        with torch.no_grad():
            cxi, cxg, _ = _tensors(ca, device)
            if args.gen_zero:
                cxg = torch.zeros_like(cxg)
            prob = torch.sigmoid(model(cxi, cxg)["logit"]).cpu().numpy()
        m = metrics.evaluate(prob, ca.y, ca.condition_id, ca.lineage)
        if m["cond_auroc"] > best_auroc:
            best_auroc, best_state, bad = m["cond_auroc"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d}  calib {metrics.format_metrics(m)}")
        if bad >= args.patience:
            print(f"early stop at ep {ep} (best calib cond_auroc={best_auroc:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Test metrics + dump teacher outputs for every split (for the student).
    out_dir = Path(cfg.paths.processed_dir) / "teacher" / (target + args.out_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", tr), ("calib", ca), ("test", te)):
        with torch.no_grad():
            xg_eval = torch.from_numpy(d.x_gen).to(device)
            if args.gen_zero:
                xg_eval = torch.zeros_like(xg_eval)
            o = model(torch.from_numpy(d.x_img).to(device), xg_eval)
            logit = o["logit"].cpu().numpy()
            rep = o["rep"].cpu().numpy()
            img_rep = o["img_rep"].cpu().numpy()
            img_logit = o["img_logit"].cpu().numpy()
        np.savez(out_dir / f"teacher_{name}.npz", logit=logit, rep=rep,
                 img_rep=img_rep, img_logit=img_logit, series_url=d.series_url)
        if name == "test":
            m = metrics.evaluate(torch.sigmoid(torch.from_numpy(logit)).numpy(),
                                 d.y, d.condition_id, d.lineage)
            mi = metrics.evaluate(torch.sigmoid(torch.from_numpy(img_logit)).numpy(),
                                  d.y, d.condition_id, d.lineage)
            print(f"\n=== TEACHER fused test ===\n{metrics.format_metrics(m)}")
            print(f"=== TEACHER image-head test ===\n{metrics.format_metrics(mi)}")
            print(f"teacher fused soft-prob std (train) = "
                  f"{float(torch.sigmoid(torch.from_numpy(np.load(out_dir/'teacher_train.npz')['logit'])).std()):.4f}")

    torch.save({"state_dict": best_state,
                "img_dim": tr.x_img.shape[1], "gen_dim": tr.x_gen.shape[1],
                "hidden": args.hidden, "rep_dim": args.rep_dim,
                "feature_names": spec.feature_names},
               out_dir / "teacher.pt")
    print(f"saved {out_dir/'teacher.pt'}")


if __name__ == "__main__":
    main()
