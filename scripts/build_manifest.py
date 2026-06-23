#!/usr/bin/env python
"""Build the paired manifest + splits from the raw TB Portals tables.

Reads genomics + CXR metadata straight from their DUA zips (no raw copy), derives
condition-level labels, joins to the paired CXR subset, assigns patient-grouped
stratified splits, and writes the result to ``data/processed/``.

Run:  python scripts/build_manifest.py [--config configs/data.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pusketb.config import load_config
from pusketb.data import labels, manifest, splits, sources


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    proc = Path(cfg.paths.processed_dir)
    interim = Path(cfg.paths.interim_dir)
    proc.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    print("[1/5] reading raw tables from DUA zips ...")
    genomics = sources.read_genomics(cfg.paths.genomics_zip)
    cxr_meta = sources.read_cxr_meta(cfg.paths.cxr_meta_zip)
    image_manifest = sources.read_image_manifest(cfg.paths.cxr_image_dir)
    print(f"      genomics specimens={len(genomics)}  cxr series={len(cxr_meta)}  "
          f"image-manifest rows={len(image_manifest)}")

    print("[2/5] deriving condition-level labels ...")
    cond = labels.build_condition_labels(genomics, cfg)
    cond.to_parquet(interim / "condition_labels.parquet", index=False)
    print(f"      labelled conditions={len(cond)}  "
          f"rif+ rate={cond['rif_resistant'].mean():.3f}")

    print("[3/5] joining imaging <-> labels (paired subset) ...")
    man = manifest.build_manifest(cxr_meta, image_manifest, cond, cfg)
    print(f"      paired image rows={len(man)}  "
          f"paired conditions={man['condition_id'].nunique()}  "
          f"unmapped-zip rows={man.attrs.get('n_missing_zip', 'NA')}")

    print("[4/5] assigning patient-grouped stratified splits ...")
    man = splits.assign_splits(man, cfg)

    print("[5/5] writing outputs ...")
    man.to_parquet(proc / "manifest.parquet", index=False)
    summary = splits.split_summary(man, cfg)
    summary.to_csv(proc / "split_summary.csv", index=False)

    # Lineage x target crosstab (the C2 reality check).
    conds = man.drop_duplicates(subset=["condition_id"])
    xt = pd.crosstab(conds["lineage_group"], conds["rif_resistant"])
    xt.to_csv(proc / "lineage_label_crosstab.csv")

    print("\n=== split summary ===")
    print(summary.to_string(index=False))
    print("\n=== lineage_group x rif_resistant (conditions) ===")
    print(xt.to_string())
    print(f"\nwrote: {proc/'manifest.parquet'}")


if __name__ == "__main__":
    main()
