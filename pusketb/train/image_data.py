"""Image dataset for end-to-end (mobile) student training with distillation.

Yields a normalized 3-channel CXR tensor plus the hard label and, when distilling,
the teacher's per-image soft logit and image-pathway representation (keyed by the
DICOM series URL). Light augmentation is applied on the train split only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from pusketb.data import dicom_extract

# RAD-DINO PNGs are grayscale; replicate to 3 channels and use ImageNet stats
# (timm backbones are ImageNet-pretrained).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(7),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ])
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


def rows_for_split(cfg, manifest: pd.DataFrame, split: str, target: str) -> pd.DataFrame:
    """Rows with an existing PNG and a defined target for ``split``."""
    df = manifest[manifest["split"] == split].copy()
    df = df[df[target].notna()]
    df["png"] = [
        str(dicom_extract.png_path(cfg.paths.images_dir, c, u))
        for c, u in zip(df["condition_id"], df["series_instance_content_url"])
    ]
    df = df[df["png"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    df["y"] = df[target].astype(np.float32)
    return df


class CXRDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, img_size: int, train: bool,
                 teacher_logit: dict[str, float] | None = None,
                 teacher_rep: dict[str, np.ndarray] | None = None):
        self.rows = rows.reset_index(drop=True)
        self.tf = build_transforms(img_size, train)
        self.tl = teacher_logit
        self.tr = teacher_rep

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = self.tf(Image.open(r["png"]))
        item = {"image": img, "y": torch.tensor(r["y"], dtype=torch.float32), "idx": i}
        if self.tl is not None:
            url = r["series_instance_content_url"]
            item["t_logit"] = torch.tensor(self.tl.get(url, 0.0), dtype=torch.float32)
            item["t_rep"] = torch.tensor(self.tr[url], dtype=torch.float32)
        return item
