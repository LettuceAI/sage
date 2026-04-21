"""Multi-label loss with per-class positive weighting and focal reshaping.

For rare / high-cost categories (``self_harm``, ``grooming``, ``sexual_minors``)
we want to:
1. Up-weight the positive class so the loss does not ignore them (pos_weight).
2. Down-weight easy examples so the model keeps learning on hard cases
   (focal reshaping, γ > 0).

Common categories (``nsfw``, ``violence``, ``harassment``, ``hate_speech``) do
not need focal; plain BCE with pos_weight is fine.

The :class:`SageLoss` below implements this per-class.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from sage.schema import CATEGORIES, Category


# Default per-class gamma. 0.0 means vanilla BCE for that class.
DEFAULT_FOCAL_GAMMA: dict[Category, float] = {
    Category.NSFW: 0.0,
    Category.VIOLENCE: 0.0,
    Category.HARASSMENT: 0.0,
    Category.HATE_SPEECH: 0.0,
    Category.SELF_HARM: 2.0,
    Category.GROOMING: 2.0,
    Category.SEXUAL_MINORS: 2.0,
}


class SageLoss(nn.Module):
    """Per-class BCE-with-logits, optionally reshaped as focal loss.

    Args:
        pos_weight: 1-D tensor of shape ``(num_classes,)`` — positive-class
            weights. Recommended: ``(num_neg / num_pos)`` per class, clamped
            to a sane upper bound (e.g. 20).
        gamma: 1-D tensor of shape ``(num_classes,)`` — focal γ per class.
            ``γ = 0`` gives vanilla weighted BCE.
    """

    def __init__(
        self,
        pos_weight: Tensor | None = None,
        gamma: Tensor | None = None,
    ) -> None:
        super().__init__()
        num_classes = len(CATEGORIES)
        if pos_weight is None:
            pos_weight = torch.ones(num_classes)
        if gamma is None:
            gamma = torch.tensor([DEFAULT_FOCAL_GAMMA[c] for c in CATEGORIES], dtype=torch.float32)
        assert pos_weight.shape == (num_classes,)
        assert gamma.shape == (num_classes,)
        self.register_buffer("pos_weight", pos_weight.float())
        self.register_buffer("gamma", gamma.float())

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """``logits``: (B, C). ``targets``: (B, C) floats in [0, 1]."""
        # Per-element BCE with pos_weight (numerically stable).
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )  # (B, C)

        # Focal reshaping: (1 - p_t)^gamma where p_t is the probability of the
        # ground-truth class.
        # For each element, p_t = σ(logits) if target=1, else 1 - σ(logits).
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        gamma = self.gamma.to(logits.device)
        focal_factor = (1.0 - p_t).pow(gamma)  # broadcasts over batch

        loss = focal_factor * bce  # (B, C)
        return loss.mean()


def compute_pos_weights(
    label_counts: dict[Category, int],
    total: int,
    *,
    max_weight: float = 20.0,
) -> Tensor:
    """Compute per-class pos_weight from label positive counts.

    ``pos_weight = num_negatives / num_positives`` per class, clamped to
    ``max_weight`` to prevent runaway gradients on extremely rare classes.
    """
    weights: list[float] = []
    for c in CATEGORIES:
        n_pos = label_counts.get(c, 0)
        n_neg = max(total - n_pos, 1)
        if n_pos <= 0:
            w = max_weight
        else:
            w = min(max_weight, n_neg / n_pos)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)
