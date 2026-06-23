#!/usr/bin/env python
"""Decode CXR DICOMs referenced by the manifest into normalized PNGs.

Resumable and shardable. Each ``*_NN_of_20.zip`` part is opened once. Use ``--part``
to process a single archive (run several in parallel), or ``--limit`` for a smoke test.

Run:  python scripts/extract_images.py [--part 01_of_20] [--limit 50]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from pusketb.config import load_config
from pusketb.data import dicom_extract


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--part", default=None,
                    help="only this zip part, e.g. '01_of_20' (substring match on zip_file)")
    ap.add_argument("--split", default=None, choices=["train", "calib", "test"],
                    help="only rows in this split")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke test)")
    ap.add_argument("--no-skip", action="store_true", help="re-decode existing PNGs")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    man = pd.read_parquet(Path(cfg.paths.processed_dir) / "manifest.parquet")
    if args.part:
        man = man[man["zip_file"].str.contains(args.part, na=False)]
    if args.split:
        man = man[man["split"] == args.split]
    if args.limit:
        man = man.head(args.limit)
    print(f"extracting {len(man)} series "
          f"(parts: {sorted(man['zip_file'].dropna().unique())[:3]}{' ...' if man['zip_file'].nunique() > 3 else ''})")

    counts: Counter[str] = Counter()
    errors: list[str] = []
    for i, (url, status, msg) in enumerate(
        dicom_extract.iter_extract(man, cfg, skip_existing=not args.no_skip), 1
    ):
        counts[status] += 1
        if status == "error":
            errors.append(f"{url}: {msg}")
        if i % 200 == 0:
            print(f"  {i}/{len(man)}  ok={counts['ok']} skip={counts['skip']} err={counts['error']}")

    print(f"\ndone: ok={counts['ok']} skip={counts['skip']} error={counts['error']}")
    if errors:
        print("first errors:")
        for e in errors[:10]:
            print("  -", e)


if __name__ == "__main__":
    main()
