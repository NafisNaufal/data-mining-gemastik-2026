"""Evaluation metrics, including the lineage-stratified breakdown for contribution C2.

Predictions are aggregated to patient (condition) level before scoring, and AUROC is
reported overall and within the L2 / L4 strata (plus the lineage-resistance prevalence
gap that a shortcut-learning X-ray model could exploit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auroc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def aggregate_to_condition(prob, y, condition_id, lineage):
    """Mean probability per condition; label/lineage taken as the condition's value."""
    df = pd.DataFrame({"prob": prob, "y": y, "condition_id": condition_id, "lineage": lineage})
    g = df.groupby("condition_id").agg(prob=("prob", "mean"), y=("y", "first"),
                                       lineage=("lineage", "first")).reset_index()
    return g


def evaluate(prob, y, condition_id, lineage, eval_groups=("L2", "L4")) -> dict:
    """Return overall + per-lineage metrics at both image and condition level."""
    res: dict[str, float] = {}
    # Image-level.
    res["img_auroc"] = _safe_auroc(y, prob)
    res["img_auprc"] = float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")

    # Condition-level (clinical unit).
    g = aggregate_to_condition(prob, y, condition_id, lineage)
    res["cond_n"] = int(len(g))
    res["cond_auroc"] = _safe_auroc(g["y"], g["prob"])
    res["cond_auprc"] = float(average_precision_score(g["y"], g["prob"])) if g["y"].nunique() > 1 else float("nan")
    res["cond_acc@0.5"] = float(((g["prob"] >= 0.5).astype(int) == g["y"]).mean())

    # Lineage-stratified (C2).
    for grp in eval_groups:
        sub = g[g["lineage"] == grp]
        res[f"auroc_{grp}"] = _safe_auroc(sub["y"], sub["prob"]) if len(sub) else float("nan")
        res[f"n_{grp}"] = int(len(sub))
        res[f"prev_{grp}"] = float(sub["y"].mean()) if len(sub) else float("nan")
    return res


def format_metrics(m: dict) -> str:
    keys = ["cond_n", "cond_auroc", "cond_auprc", "cond_acc@0.5", "img_auroc",
            "auroc_L2", "n_L2", "prev_L2", "auroc_L4", "n_L4", "prev_L4"]
    return "  ".join(f"{k}={m[k]:.4f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
                     for k in keys if k in m)
