import torch

from sage.loss import DEFAULT_FOCAL_GAMMA, SageLoss, compute_pos_weights
from sage.schema import CATEGORIES, Category
from training.train import _extract_layer_index, build_param_groups


def test_pos_weight_for_rare_class_is_clamped():
    # 1 positive in 1M → raw weight would be 1M-1, clamp to max_weight
    counts = {Category.GROOMING: 1}
    w = compute_pos_weights(counts, total=1_000_000, max_weight=20.0)
    idx = list(CATEGORIES).index(Category.GROOMING)
    assert w[idx].item() == 20.0


def test_pos_weight_for_missing_class_is_max_weight():
    w = compute_pos_weights({}, total=100, max_weight=15.0)
    assert torch.allclose(w, torch.full_like(w, 15.0))


def test_loss_goes_down_when_predictions_match_targets():
    torch.manual_seed(0)
    num_c = len(CATEGORIES)
    loss = SageLoss(
        pos_weight=torch.ones(num_c),
        gamma=torch.zeros(num_c),
    )
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    good = torch.tensor([[5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0]])
    bad = torch.tensor([[-5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0]])
    assert loss(good, targets).item() < loss(bad, targets).item()


def test_focal_downweights_easy_examples():
    num_c = len(CATEGORIES)
    # All classes have gamma=2.
    gamma = torch.full((num_c,), 2.0)
    pos_w = torch.ones(num_c)
    l_focal = SageLoss(pos_weight=pos_w, gamma=gamma)
    l_bce = SageLoss(pos_weight=pos_w, gamma=torch.zeros(num_c))
    targets = torch.ones(1, num_c)
    confident_correct = torch.full((1, num_c), 6.0)  # σ(6)≈0.998 — very easy positive
    # Focal should produce a much smaller loss on easy examples
    assert l_focal(confident_correct, targets).item() < l_bce(confident_correct, targets).item()


def test_default_gamma_aligns_with_schema_order():
    # The list version that SageLoss builds internally must align with CATEGORIES.
    gammas = [DEFAULT_FOCAL_GAMMA[c] for c in CATEGORIES]
    # Rare-class positions should be > 0
    for c, g in zip(CATEGORIES, gammas):
        if c in (Category.SELF_HARM, Category.GROOMING, Category.SEXUAL_MINORS):
            assert g > 0
        else:
            assert g == 0


def test_extract_layer_index_parses_common_names():
    assert _extract_layer_index("encoder.layer.0.attention.self.query.weight") == 0
    assert _extract_layer_index("encoder.layer.11.output.dense.bias") == 11
    assert _extract_layer_index("embeddings.word_embeddings.weight") is None
    assert _extract_layer_index("pooler.dense.weight") is None


def test_build_param_groups_produces_four_buckets():
    """Sanity: build on a minimal stub module that mimics the real structure.

    We avoid actually instantiating Jina v2 by crafting a fake encoder + model.
    """
    from torch import nn

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Linear(10, 10)
            self.layer = nn.ModuleList([nn.Linear(10, 10) for _ in range(12)])
            self.pooler = nn.Linear(10, 10)

        def named_parameters(self, prefix: str = "", recurse: bool = True):
            return super().named_parameters(prefix=prefix, recurse=recurse)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FakeEncoder()
            self.classifier = nn.Linear(10, 7)

    m = FakeModel()
    groups = build_param_groups(m, num_layers=12)
    assert len(groups) == 4
    # Each group must have at least one parameter (except possibly head if no
    # classifier — but we have one).
    for g in groups:
        assert len(list(g["params"])) > 0
    # The four groups must have distinct LRs
    lrs = [g["lr"] for g in groups]
    assert len(set(lrs)) == 4
