"""Tests for per-example observation masking across the pipeline."""

import torch

from sage.loss import SageLoss, compute_pos_weights
from sage.schema import CATEGORIES, Category
from training.data.example import ALL_CATEGORIES, Example


# ---------------------------------------------------------------------------
# Example-level
# ---------------------------------------------------------------------------
def test_default_observed_is_all_categories():
    ex = Example.from_text("hi", labels={Category.HARASSMENT: 0.9})
    assert ex.observed == ALL_CATEGORIES
    assert ex.observation_mask() == [1.0] * len(CATEGORIES)


def test_partial_observed_produces_partial_mask():
    ex = Example.from_text(
        "hi",
        labels={Category.HATE_SPEECH: 0.8},
        observed=frozenset({Category.HATE_SPEECH}),
    )
    mask = ex.observation_mask()
    assert sum(mask) == 1.0
    idx = list(CATEGORIES).index(Category.HATE_SPEECH)
    assert mask[idx] == 1.0


def test_is_negative_only_considers_observed_labels():
    # MHS-style: observed only hate_speech, labels empty → clean negative for
    # the one observed category. Unobserved positions do not affect is_negative.
    ex = Example.from_text(
        "benign",
        labels={},
        observed=frozenset({Category.HATE_SPEECH}),
    )
    assert ex.is_negative()


# ---------------------------------------------------------------------------
# Loss-level
# ---------------------------------------------------------------------------
def test_loss_masks_unobserved_positions():
    num_c = len(CATEGORIES)
    loss_fn = SageLoss(
        pos_weight=torch.ones(num_c),
        gamma=torch.zeros(num_c),
    )
    # One example, wildly wrong on positions 0..3, correct on 4..6.
    logits = torch.tensor([[-5.0, -5.0, -5.0, -5.0, 5.0, 5.0, 5.0]])
    targets = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    mask_all = torch.ones_like(targets)
    # Observing only the correct positions should produce near-zero loss.
    mask_correct_only = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    loss_all = loss_fn(logits, targets, observed_mask=mask_all).item()
    loss_correct = loss_fn(logits, targets, observed_mask=mask_correct_only).item()

    assert loss_correct < 0.1
    assert loss_all > 1.0


def test_loss_all_unobserved_yields_zero():
    """If no positions are observed, loss is zero rather than NaN."""
    num_c = len(CATEGORIES)
    loss_fn = SageLoss(
        pos_weight=torch.ones(num_c),
        gamma=torch.zeros(num_c),
    )
    logits = torch.randn(2, num_c)
    targets = torch.randint(0, 2, (2, num_c)).float()
    mask = torch.zeros_like(targets)
    loss = loss_fn(logits, targets, observed_mask=mask)
    assert torch.isfinite(loss)
    assert abs(loss.item()) < 1e-6


def test_pos_weights_use_observed_count_not_total():
    # Rare class: 10 positives out of 100 observed, but dataset has 1000 rows
    # overall. pos_weight should be based on observed negatives (90), not on
    # corpus size (990). Expected ≈ 90 / 10 = 9, not 99.
    label_counts = {Category.GROOMING: 10}
    observed_counts = {c: 0 for c in CATEGORIES}
    observed_counts[Category.GROOMING] = 100
    w = compute_pos_weights(
        label_counts,
        total=1000,
        observed_counts=observed_counts,
        max_weight=50.0,
    )
    idx = list(CATEGORIES).index(Category.GROOMING)
    assert abs(w[idx].item() - 9.0) < 0.01
