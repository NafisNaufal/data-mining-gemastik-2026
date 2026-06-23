"""Distillation losses for cross-modal teacher -> student (contribution C1).

Total student objective:

    L = w_ce * BCE(student_logit, y)
      + w_kd * KD(student_logit, teacher_logit; T)        # Hinton soft-label transfer
      + w_feat * (1 - cos(student_distill_rep, teacher_rep))  # feature / "hint" transfer
      + w_con * InfoNCE(student_distill_rep, teacher_rep)  # optional contrastive (L_contrastive)

The feature/contrastive channels matter here because the genomics-informed teacher is
near-oracle: its soft labels carry little beyond the hard labels, but its *fused
representation* encodes lineage/co-resistance structure worth transferring into the
image-only student. ``L_contrastive`` is first on the project's scope-cut list, so it
is opt-in (``w_con=0`` by default).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(student_logit: torch.Tensor, teacher_logit: torch.Tensor, T: float = 2.0):
    """Binary soft-label KD: KL between temperature-softened teacher/student sigmoids."""
    s = torch.stack([student_logit / T, torch.zeros_like(student_logit)], dim=-1)
    t = torch.stack([teacher_logit / T, torch.zeros_like(teacher_logit)], dim=-1)
    log_p_s = F.log_softmax(s, dim=-1)
    p_t = F.softmax(t, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def feature_distill_loss(student_rep: torch.Tensor, teacher_rep: torch.Tensor):
    """Cosine-alignment hint loss (student projected into teacher rep space)."""
    return (1.0 - F.cosine_similarity(student_rep, teacher_rep.detach(), dim=-1)).mean()


def contrastive_loss(student_rep: torch.Tensor, teacher_rep: torch.Tensor, tau: float = 0.1):
    """InfoNCE aligning each student rep to its own teacher rep against in-batch negatives."""
    s = F.normalize(student_rep, dim=-1)
    t = F.normalize(teacher_rep.detach(), dim=-1)
    logits = s @ t.t() / tau
    target = torch.arange(s.size(0), device=s.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def student_loss(out_s: dict, out_t: dict, y: torch.Tensor, pos_weight: torch.Tensor | None,
                 w_ce=1.0, w_kd=1.0, w_feat=1.0, w_con=0.0, T=2.0, tau=0.1):
    """Combined student objective; returns (total, components dict)."""
    ce = F.binary_cross_entropy_with_logits(out_s["logit"], y, pos_weight=pos_weight)
    comp = {"ce": ce.item()}
    total = w_ce * ce

    if w_kd > 0 and out_t is not None:
        kd = kd_loss(out_s["logit"], out_t["logit"].detach(), T=T)
        total = total + w_kd * kd
        comp["kd"] = kd.item()
    if w_feat > 0 and out_t is not None:
        feat = feature_distill_loss(out_s["distill_rep"], out_t["rep"])
        total = total + w_feat * feat
        comp["feat"] = feat.item()
    if w_con > 0 and out_t is not None:
        con = contrastive_loss(out_s["distill_rep"], out_t["rep"], tau=tau)
        total = total + w_con * con
        comp["con"] = con.item()
    return total, comp
