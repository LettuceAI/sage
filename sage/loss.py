"""Multi-label loss with per-class positive weighting, focal reshaping, and
per-example observation masking.

For rare / high-cost categories (``self_harm``, ``grooming``, ``sexual_minors``)
we want to:
1. Up-weight the positive class so the loss does not ignore them (pos_weight).
2. Down-weight easy examples so the model keeps learning on hard cases
   (focal reshaping, γ > 0).

The observation mask accounts for the fact that each training source only
labels a subset of our seven categories. When a source did not examine a
category, that dim must contribute zero loss — otherwise the model learns
"this is negative" for every unobserved category, which is a silent false
negative and will hurt rare-class recall. See
:func:`SageLoss.forward` for the exact formulation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sage.schema import CATEGORIES, Category

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
    """Per-class BCE-with-logits, optionally reshaped as focal loss, masked by
    per-example observation.

    Args:
        pos_weight: 1-D tensor of shape ``(num_classes,)``. Positive-class
            weights, typically ``num_observed_negatives / num_positives``.
        gamma: 1-D tensor of shape ``(num_classes,)``. Focal γ per class.
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

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        observed_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the masked focal-BCE loss.

        Args:
            logits: ``(B, C)`` unnormalized predictions.
            targets: ``(B, C)`` float labels in ``[0, 1]``.
            observed_mask: optional ``(B, C)`` 0/1 mask. 1 at ``(i, c)`` means
                example ``i`` actually observed category ``c``. Unobserved
                positions contribute zero loss.
        """
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )  # (B, C)

        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        gamma = self.gamma.to(logits.device)
        focal_factor = (1.0 - p_t).pow(gamma)

        loss = focal_factor * bce  # (B, C)
        if observed_mask is not None:
            loss = loss * observed_mask
            denom = observed_mask.sum().clamp_min(1.0)
            return loss.sum() / denom
        return loss.mean()


def compute_pos_weights(
    label_counts: dict[Category, int],
    total: int,
    *,
    observed_counts: dict[Category, int] | None = None,
    max_weight: float = 20.0,
) -> Tensor:
    """Compute per-class ``pos_weight`` from label positive counts.

    When ``observed_counts`` is provided, the denominator for each category is
    its observed count rather than the corpus total. This is important when
    some sources did not examine a category: we want
    ``pos_weight ≈ (observed_negatives / positives)``, not
    ``(corpus_size / positives)``, because unobserved positions never
    contribute to the loss.
    """
    weights: list[float] = []
    for c in CATEGORIES:
        n_pos = label_counts.get(c, 0)
        n_obs = observed_counts.get(c, 0) if observed_counts is not None else total
        n_neg = max(n_obs - n_pos, 1)
        if n_pos <= 0:
            w = max_weight
        else:
            w = min(max_weight, n_neg / n_pos)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)
