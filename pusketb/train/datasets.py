"""Assemble per-split training tensors from cached embeddings + genomic features.

Samples are at the **image (series)** level — each CXR is one example carrying its
patient's genomic features and label. Evaluation aggregates predictions back to the
**patient (condition)** level (mean probability over a patient's images), which is the
clinically meaningful unit and matches the patient-grouped splits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pusketb.data import genomics_features as gf


@dataclass
class SplitData:
    x_img: np.ndarray          # (N, img_dim) float32
    x_gen: np.ndarray          # (N, gen_dim) float32
    y: np.ndarray              # (N,) float32
    condition_id: np.ndarray   # (N,) str
    lineage: np.ndarray        # (N,) str
    series_url: np.ndarray     # (N,) str  — key to the source DICOM / PNG


def _load_embeddings(emb_dir: Path, split: str):
    npy = emb_dir / f"raddino_{split}.npy"
    idx = emb_dir / f"raddino_{split}_index.parquet"
    if not npy.exists() or not idx.exists():
        raise FileNotFoundError(
            f"missing cached embeddings for split '{split}' ({npy.name}); "
            f"run: python scripts/cache_embeddings.py --split {split}"
        )
    return np.load(npy), pd.read_parquet(idx)


def load_all(cfg: SimpleNamespace, splits=("train", "calib", "test"),
             target: str | None = None) -> dict[str, SplitData]:
    """Load every requested split, fitting the genomics spec on train conditions only.

    ``target`` selects the label column (defaults to ``cfg.labels.target``); labels are
    joined from the manifest by series URL, so switching targets needs no re-caching.
    Rows whose target is undefined (NaN, e.g. unknown outcome) are dropped.
    """
    proc = Path(cfg.paths.processed_dir)
    emb_dir = Path(cfg.paths.embeddings_dir)
    cond = pd.read_parquet(Path(cfg.paths.interim_dir) / "condition_labels.parquet")
    manifest = pd.read_parquet(proc / "manifest.parquet")
    target = target or cfg.labels.target

    key = "series_instance_content_url"
    label_map = manifest.set_index(key)[target]
    lin_map = manifest.set_index(key)["lineage_group"]

    # The privileged clinical block (LUPI) is enabled for the targets listed in config
    # (treatment outcome). Clinical fields live in the manifest; merge them onto the
    # condition-level table (clinical attributes are condition-constant in TB Portals).
    clinical_targets = set(getattr(cfg.labels, "clinical_targets", []) or [])
    include_clinical = target in clinical_targets
    if include_clinical:
        clin = (manifest.drop_duplicates("condition_id")
                .set_index("condition_id")[gf.CLINICAL_SOURCE_COLS])
        cond = cond.merge(clin, on="condition_id", how="left")

    train_ids = manifest.loc[manifest["split"] == "train", "condition_id"].unique().tolist()
    spec = gf.fit_spec(cond, cfg, train_ids, target=target, include_clinical=include_clinical)
    feats = gf.transform(cond, spec).set_index("condition_id")
    feat_cols = list(feats.columns)

    out: dict[str, SplitData] = {}
    for split in splits:
        emb, idx = _load_embeddings(emb_dir, split)
        if len(emb) != len(idx):
            raise ValueError(f"embedding/index length mismatch for {split}")
        y = idx[key].map(label_map).to_numpy(dtype=float)
        lineage = idx[key].map(lin_map).astype(str).to_numpy()
        keep = ~np.isnan(y)  # drop undefined-target rows
        gen = feats.reindex(idx["condition_id"].values)[feat_cols].to_numpy(np.float32)
        out[split] = SplitData(
            x_img=emb.astype(np.float32)[keep],
            x_gen=gen[keep],
            y=y[keep].astype(np.float32),
            condition_id=idx["condition_id"].astype(str).to_numpy()[keep],
            lineage=lineage[keep],
            series_url=idx[key].astype(str).to_numpy()[keep],
        )
    out["_spec"] = spec  # type: ignore[assignment]
    return out
