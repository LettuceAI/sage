"""Focal BCE loss with per-class pos_weight and per-example observation mask."""

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
    def __init__(self, pos_weight: Tensor | None = None, gamma: Tensor | None = None) -> None:
        super().__init__()
        c = len(CATEGORIES)
        if pos_weight is None:
            pos_weight = torch.ones(c)
        if gamma is None:
            gamma = torch.tensor([DEFAULT_FOCAL_GAMMA[k] for k in CATEGORIES], dtype=torch.float32)
        assert pos_weight.shape == (c,)
        assert gamma.shape == (c,)
        self.register_buffer("pos_weight", pos_weight.float())
        self.register_buffer("gamma", gamma.float())

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        observed_mask: Tensor | None = None,
    ) -> Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        loss = (1.0 - p_t).pow(self.gamma.to(logits.device)) * bce
        if observed_mask is not None:
            loss = loss * observed_mask
            return loss.sum() / observed_mask.sum().clamp_min(1.0)
        return loss.mean()


def compute_pos_weights(
    label_counts: dict[Category, int],
    total: int,
    *,
    observed_counts: dict[Category, int] | None = None,
    max_weight: float = 20.0,
) -> Tensor:
    weights: list[float] = []
    for c in CATEGORIES:
        n_pos = label_counts.get(c, 0)
        n_obs = observed_counts.get(c, 0) if observed_counts is not None else total
        n_neg = max(n_obs - n_pos, 1)
        w = max_weight if n_pos <= 0 else min(max_weight, n_neg / n_pos)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)
