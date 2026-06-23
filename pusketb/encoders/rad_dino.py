"""RAD-DINO image encoder (microsoft/rad-dino) — frozen feature extractor.

RAD-DINO is a ViT-B/14 trained on chest X-rays via DINOv2. Here we use it as the
*teacher's* frozen image backbone: given a normalized CXR PNG it returns a pooled
embedding (CLS) and, optionally, patch tokens. End-to-end fine-tuning later just
swaps ``torch.no_grad`` for a trainable wrapper around the same model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "microsoft/rad-dino"


class RadDino:
    """Frozen RAD-DINO encoder with a simple ``embed`` API over PIL images."""

    def __init__(self, device: str | None = None, model_id: str = MODEL_ID,
                 dtype: torch.dtype = torch.float32):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.model.eval().to(self.device)
        self.dtype = dtype

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def embed(self, images: list[Image.Image], pooling: str = "rich",
              return_patches: bool = False):
        """Embed a batch of PIL images.

        pooling="cls"  -> CLS token only (768-d).
        pooling="rich" -> [CLS ‖ mean-patch ‖ max-patch] (2304-d). Captures focal
                          pathology (cavitation) that the global CLS vector misses.
        """
        rgb = [im.convert("RGB") for im in images]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        cls = getattr(out, "pooler_output", None)
        if cls is None:
            cls = out.last_hidden_state[:, 0]
        patches = out.last_hidden_state[:, 1:]            # (B, N, 768)
        if pooling == "cls":
            pooled = cls
        else:
            pooled = torch.cat([cls, patches.mean(1), patches.amax(1)], dim=-1)
        result = {"pooled": pooled.float().cpu().numpy(),
                  "cls": cls.float().cpu().numpy()}
        if return_patches:
            result["patches"] = patches.float().cpu().numpy()
        return result


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")
