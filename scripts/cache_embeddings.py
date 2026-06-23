#!/usr/bin/env python
"""Precompute & cache frozen RAD-DINO embeddings for extracted CXR PNGs.

Lets us prototype the multimodal teacher, the distilled student heads, the
cost-sensitive ranking loss, and conformal calibration without paying the encoder
forward pass every step. Writes one .npy of pooled embeddings plus an aligned index
parquet (series_url, condition_id, split, target) per shard.

Run:  python scripts/cache_embeddings.py [--split train] [--batch-size 64] [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pusketb.config import load_config
from pusketb.data import dicom_extract


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None, choices=["train", "calib", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    from pusketb.encoders.rad_dino import RadDino, load_image

    man = pd.read_parquet(Path(cfg.paths.processed_dir) / "manifest.parquet")
    if args.split:
        man = man[man["split"] == args.split]
    if args.limit:
        man = man.head(args.limit)

    images_dir = cfg.paths.images_dir
    # Resolve PNG paths; keep only rows whose image has been extracted.
    man = man.copy()
    man["png"] = [
        str(dicom_extract.png_path(images_dir, c, u))
        for c, u in zip(man["condition_id"], man["series_instance_content_url"])
    ]
    present = man[man["png"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    print(f"{len(present)}/{len(man)} rows have extracted PNGs")
    if present.empty:
        print("nothing to embed — run scripts/extract_images.py first")
        return

    enc = RadDino()
    out_dir = Path(cfg.paths.embeddings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.split or "all"

    embs: list[np.ndarray] = []
    for i in range(0, len(present), args.batch_size):
        batch = present.iloc[i:i + args.batch_size]
        imgs = [load_image(p) for p in batch["png"]]
        embs.append(enc.embed(imgs)["pooled"])
        print(f"  embedded {min(i + args.batch_size, len(present))}/{len(present)}")
    emb = np.concatenate(embs, axis=0)

    np.save(out_dir / f"raddino_{tag}.npy", emb)
    keep = ["condition_id", "series_instance_content_url", "split",
            cfg.labels.target, "lineage_group", "png"]
    present[[c for c in keep if c in present.columns]].to_parquet(
        out_dir / f"raddino_{tag}_index.parquet", index=False
    )
    print(f"wrote {out_dir/f'raddino_{tag}.npy'}  shape={emb.shape} (dim={enc.dim})")


if __name__ == "__main__":
    main()
