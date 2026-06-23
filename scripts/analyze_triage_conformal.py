#!/usr/bin/env python
"""C3 (cost-sensitive triage) + C4 (conformal prediction) analysis for the student.

Loads a trained student, scores calib + test at the patient (condition) level, then:

* **C3 — triage under GeneXpert capacity M < N.** Rank patients by predicted rifampicin-
  resistance probability; for each capacity fraction c, test the top fraction and report
  the share of truly resistant patients caught (sensitivity) vs a random-allocation
  baseline (= c). This quantifies the clinical value of even a moderate-AUROC model.
* **C4 — split-conformal prediction sets (LAC).** Calibrate a threshold on the calib
  split so test prediction sets achieve marginal coverage >= 1 - alpha; report empirical
  coverage, mean set size, and singleton rate (decisive predictions).

Run:  python scripts/analyze_triage_conformal.py [--mode baseline] [--alpha 0.1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pusketb.config import load_config
from pusketb.models.heads import StudentNet
from pusketb.train import datasets, metrics


def condition_probs(model, d, device):
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(d.x_img).to(device))["logit"]).cpu().numpy()
    g = metrics.aggregate_to_condition(p, d.y, d.condition_id, d.lineage)
    return g["prob"].to_numpy(), g["y"].to_numpy().astype(int), g["lineage"].to_numpy()


def triage_curve(prob, y, fractions=(0.1, 0.2, 0.3, 0.4, 0.5)):
    order = np.argsort(-prob)
    y_sorted = y[order]
    total_pos = max(int(y.sum()), 1)
    n = len(y)
    rows = []
    for c in fractions:
        m = max(1, int(round(c * n)))
        caught = int(y_sorted[:m].sum())
        rows.append({
            "capacity": c,
            "tested": m,
            "sens_model": caught / total_pos,        # share of resistant caught
            "sens_random": c,                         # random allocation baseline
            "precision": caught / m,
            "lift": (caught / total_pos) / c,
        })
    return rows


def conformal_lac(p_cal, y_cal, p_test, y_test, alpha=0.1):
    """Binary split-conformal (LAC): nonconformity = 1 - p(true label)."""
    p_cal2 = np.stack([1 - p_cal, p_cal], axis=1)   # P(y=0), P(y=1)
    scores = 1.0 - p_cal2[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    qlevel = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores, qlevel, method="higher")

    p_test2 = np.stack([1 - p_test, p_test], axis=1)
    include = (1.0 - p_test2) <= qhat          # (N,2) bool: which labels in the set
    covered = include[np.arange(len(y_test)), y_test]
    set_size = include.sum(axis=1)
    return {
        "alpha": alpha, "qhat": float(qhat),
        "coverage": float(covered.mean()),
        "mean_set_size": float(set_size.mean()),
        "singleton_rate": float((set_size == 1).mean()),
        "empty_rate": float((set_size == 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--mode", default="baseline", choices=["baseline", "distill"])
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = args.target or cfg.labels.target

    data = datasets.load_all(cfg, target=target); data.pop("_spec")
    ca, te = data["calib"], data["test"]

    ckpt = torch.load(Path(cfg.paths.processed_dir) / "student" / target / f"student_{args.mode}.pt",
                      map_location=device)
    model = StudentNet(ckpt["img_dim"], hidden=ckpt["hidden"], rep_dim=ckpt["rep_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    p_cal, y_cal, _ = condition_probs(model, ca, device)
    p_te, y_te, lin_te = condition_probs(model, te, device)

    print(f"=== C3: cost-sensitive triage (student {args.mode}, test N={len(y_te)}, "
          f"resistant={int(y_te.sum())}) ===")
    print(f"{'cap':>5} {'tested':>7} {'sens_model':>11} {'sens_rand':>10} {'precision':>10} {'lift':>6}")
    for r in triage_curve(p_te, y_te):
        print(f"{r['capacity']:>5.0%} {r['tested']:>7d} {r['sens_model']:>11.3f} "
              f"{r['sens_random']:>10.3f} {r['precision']:>10.3f} {r['lift']:>6.2f}")

    print(f"\n=== C4: split-conformal LAC sets (alpha={args.alpha}) ===")
    cf = conformal_lac(p_cal, y_cal, p_te, y_te, alpha=args.alpha)
    for k, v in cf.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    # Coverage by lineage (C2 x C4).
    for grp in ("L2", "L4"):
        mask = lin_te == grp
        if mask.sum():
            cfg2 = conformal_lac(p_cal, y_cal, p_te[mask], y_te[mask], alpha=args.alpha)
            print(f"  [{grp}] coverage={cfg2['coverage']:.4f} set_size={cfg2['mean_set_size']:.4f} n={int(mask.sum())}")


if __name__ == "__main__":
    main()
