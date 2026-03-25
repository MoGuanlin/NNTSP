# src/models/losses.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossOut:
    loss: Tensor
    parts: Dict[str, Tensor]


def _check_same_shape(a: Tensor, b: Tensor, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a}.shape={tuple(a.shape)} must equal {name_b}.shape={tuple(b.shape)}")


def masked_bce_with_logits(
    logit: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    pos_weight: Optional[float] = None,
) -> Tensor:
    """
    Binary cross-entropy with logits over a masked subset.

    Args:
      logit/target: same shape, float
      mask: same shape, bool (True = valid)
      pos_weight: optional scalar for class imbalance (weight for positive class)

    Returns:
      mean BCE over masked entries, or 0 if no valid entries.
    """
    _check_same_shape(logit, target, "logit", "target")
    _check_same_shape(logit, mask, "logit", "mask")

    m = mask.bool()
    if not m.any().item():
        return logit.new_tensor(0.0)

    if pos_weight is None:
        loss = F.binary_cross_entropy_with_logits(logit[m], target[m], reduction="mean")
    else:
        pw = logit.new_tensor(float(pos_weight))
        loss = F.binary_cross_entropy_with_logits(logit[m], target[m], reduction="mean", pos_weight=pw)

    if not torch.isfinite(loss).item():
        raise RuntimeError("masked_bce_with_logits produced NaN/Inf.")
    return loss


def masked_ce_with_logits(
    logit: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Multi-class cross-entropy over a masked subset.

    Args:
      logit:  [N,C] float
      target: [N] long
      mask:   [N] bool
    """
    if logit.dim() != 2:
        raise ValueError(f"logit must be [N,C], got {tuple(logit.shape)}")
    if target.shape != logit.shape[:1]:
        raise ValueError(f"target must be [N], got {tuple(target.shape)} for logit {tuple(logit.shape)}")
    if mask.shape != target.shape:
        raise ValueError(f"mask must match target shape, got {tuple(mask.shape)} vs {tuple(target.shape)}")

    m = mask.bool() & (target >= 0)
    if not m.any().item():
        return logit.new_tensor(0.0)

    loss = F.cross_entropy(logit[m], target[m].long(), reduction="mean")
    if not torch.isfinite(loss).item():
        raise RuntimeError("masked_ce_with_logits produced NaN/Inf.")
    return loss


# -----------------------------
# Bottom-up (or local token) losses
# -----------------------------
def dp_token_losses(
    *,
    cross_logit: Tensor,
    y_cross: Tensor,
    m_cross: Tensor,
    iface_logit: Optional[Tensor] = None,
    y_iface: Optional[Tensor] = None,
    m_iface: Optional[Tensor] = None,
    w_iface: float = 0.1,
    pos_weight_cross: Optional[float] = None,
) -> LossOut:
    """
    Legacy / bottom-up style token loss:
      - cross: primary supervision channel
      - iface: optional auxiliary

    Shapes are arbitrary as long as they match pairwise.

    Returns:
      LossOut(loss_total, parts)
    """
    L_cross = masked_bce_with_logits(cross_logit, y_cross, m_cross, pos_weight=pos_weight_cross)
    parts: Dict[str, Tensor] = {"loss_cross": L_cross}
    loss = L_cross

    if iface_logit is not None and y_iface is not None and m_iface is not None and float(w_iface) > 0:
        L_iface = masked_bce_with_logits(iface_logit, y_iface, m_iface, pos_weight=None)
        parts["loss_iface"] = L_iface
        loss = loss + float(w_iface) * L_iface

    parts["loss_total"] = loss
    return LossOut(loss=loss, parts=parts)


# -----------------------------
# New: Top-down boundary-condition losses
# -----------------------------
def bc_child_iface_losses(
    *,
    child_iface_logit: Tensor,
    y_child_iface: Tensor,
    m_child_iface: Tensor,
    pos_weight_child: Optional[float] = None,
) -> Tensor:
    """
    Supervise boundary conditions passed from parent -> child.

    Typical shapes:
      child_iface_logit: [B,4,Ti] or [M,4,Ti]
      y_child_iface:     same shape, float in {0,1} or soft in [0,1]
      m_child_iface:     same shape, bool (valid child interfaces)

    Returns:
      mean masked BCE.
    """
    return masked_bce_with_logits(
        child_iface_logit,
        y_child_iface,
        m_child_iface,
        pos_weight=pos_weight_child,
    )


def top_down_losses(
    *,
    # required main output
    child_iface_logit: Tensor,
    y_child_iface: Tensor,
    m_child_iface: Tensor,
    # optional local aux outputs (on the same parent nodes)
    iface_logit: Optional[Tensor] = None,
    y_iface: Optional[Tensor] = None,
    m_iface: Optional[Tensor] = None,
    cross_logit: Optional[Tensor] = None,
    y_cross: Optional[Tensor] = None,
    m_cross: Optional[Tensor] = None,
    # weights
    w_child: float = 1.0,
    w_iface: float = 0.1,
    w_cross: float = 0.1,
    # imbalance controls
    pos_weight_child: Optional[float] = None,
    pos_weight_cross: Optional[float] = None,
) -> LossOut:
    """
    Unified loss for top-down stage:

    Primary:
      - child iface boundary condition logits (parent -> 4 children interfaces)

    Optional auxiliary:
      - parent local iface logits
      - parent local cross logits

    Notes:
      - This function does NOT assume any particular DP constraint (matching, degree=2, etc.).
        It is purely token-level supervision.
      - Your labeler/teacher can provide y_child_iface as hard 0/1 or soft targets.

    Returns:
      LossOut(loss_total, parts)
    """
    if float(w_child) <= 0:
        raise ValueError("w_child must be > 0 for top-down losses.")

    L_child = bc_child_iface_losses(
        child_iface_logit=child_iface_logit,
        y_child_iface=y_child_iface,
        m_child_iface=m_child_iface,
        pos_weight_child=pos_weight_child,
    )

    parts: Dict[str, Tensor] = {"loss_child_iface": L_child}
    loss = float(w_child) * L_child

    if iface_logit is not None and y_iface is not None and m_iface is not None and float(w_iface) > 0:
        L_iface = masked_bce_with_logits(iface_logit, y_iface, m_iface, pos_weight=None)
        parts["loss_iface"] = L_iface
        loss = loss + float(w_iface) * L_iface

    if cross_logit is not None and y_cross is not None and m_cross is not None and float(w_cross) > 0:
        L_cross = masked_bce_with_logits(cross_logit, y_cross, m_cross, pos_weight=pos_weight_cross)
        parts["loss_cross"] = L_cross
        loss = loss + float(w_cross) * L_cross

    parts["loss_total"] = loss
    return LossOut(loss=loss, parts=parts)


__all__ = [
    "LossOut",
    "masked_bce_with_logits",
    "masked_ce_with_logits",
    "dp_token_losses",
    "bc_child_iface_losses",
    "top_down_losses",
]
