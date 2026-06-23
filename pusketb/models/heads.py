"""Teacher and student heads for the cached-embedding prototype.

Both consume a frozen RAD-DINO image embedding (768-d). The **teacher** additionally
fuses the genomic feature vector; the **student** is image-only. Each exposes a
penultimate ``rep`` so the student can be distilled toward the teacher's *fused*
representation (the feature-distillation / "hint" channel), not only its logits.

When end-to-end fine-tuning replaces cached embeddings later, the same heads sit on
top of a trainable RAD-DINO / mobile backbone — only the feature source changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class TeacherNet(nn.Module):
    """Multimodal teacher: image embedding ⊕ genomic features → fused rep → logit.

    To make the *image pathway* a useful distillation target (and to avoid the fusion
    ignoring the image because genomics is near-oracle), the teacher carries a dedicated
    image representation ``img_rep`` with its own auxiliary classifier ``img_logit``.
    Modality dropout on the genomics input (applied in the training loop) further forces
    the fused head to rely on the image and softens the teacher's probabilities.
    """

    def __init__(self, image_dim: int, gen_dim: int, hidden: int = 512,
                 rep_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.img_proj = _mlp([image_dim, hidden, rep_dim], dropout)   # -> image rep
        self.gen_proj = _mlp([gen_dim, hidden], dropout)
        self.fuse = _mlp([rep_dim + hidden, hidden, rep_dim], dropout)
        self.classifier = nn.Linear(rep_dim, 1)       # fused head
        self.img_classifier = nn.Linear(rep_dim, 1)   # auxiliary image-only head
        self.rep_dim = rep_dim

    def forward(self, image: torch.Tensor, genomics: torch.Tensor):
        img_rep = self.img_proj(image)
        h = torch.cat([img_rep, self.gen_proj(genomics)], dim=-1)
        rep = self.fuse(h)
        return {
            "logit": self.classifier(rep).squeeze(-1),
            "rep": rep,
            "img_rep": img_rep,
            "img_logit": self.img_classifier(img_rep).squeeze(-1),
        }


class StudentNet(nn.Module):
    """X-ray-only student: image embedding → rep → logit (+ projection to teacher rep)."""

    def __init__(self, image_dim: int, hidden: int = 512, rep_dim: int = 256,
                 teacher_rep_dim: int | None = None, dropout: float = 0.3):
        super().__init__()
        self.backbone = _mlp([image_dim, hidden, rep_dim], dropout)
        self.classifier = nn.Linear(rep_dim, 1)
        # Projects the student rep into the teacher's rep space for feature distillation.
        self.distill_proj = (
            nn.Linear(rep_dim, teacher_rep_dim) if teacher_rep_dim else nn.Identity()
        )
        self.rep_dim = rep_dim

    def forward(self, image: torch.Tensor):
        rep = self.backbone(image)
        return {
            "logit": self.classifier(rep).squeeze(-1),
            "rep": rep,
            "distill_rep": self.distill_proj(rep),
        }
