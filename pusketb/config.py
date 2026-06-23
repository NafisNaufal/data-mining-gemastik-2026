"""Configuration loading.

A thin wrapper that loads ``configs/data.yaml`` into nested objects with
attribute access, resolving every path relative to the project root so scripts
can be invoked from anywhere.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Project root = parent of the ``pusketb`` package directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "data.yaml"

# Keys under ``paths:`` are resolved to absolute paths against ROOT.
_PATH_KEYS = {
    "tbportals_root", "genomics_zip", "cxr_meta_zip", "cxr_image_dir",
    "interim_dir", "processed_dir", "images_dir", "embeddings_dir",
}


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path = DEFAULT_CONFIG) -> SimpleNamespace:
    """Load the YAML config, resolving ``paths`` against the project root."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    for key, val in raw.get("paths", {}).items():
        if key in _PATH_KEYS and val is not None:
            p = Path(val)
            raw["paths"][key] = str(p if p.is_absolute() else ROOT / p)

    cfg = _to_ns(raw)
    cfg.root = str(ROOT)
    return cfg
