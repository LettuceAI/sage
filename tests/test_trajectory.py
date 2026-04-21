import random

import pytest

from sage.conversation import Conversation, Role, Turn
from sage.schema import Category
from training.data.example import Example
from training.data.trajectory import (
    BenignContextBank,
    TrajectoryConfig,
    _alternating_roles_ending_before,
    build_benign_context,
    pad_negative_with_edgy_context,
    pad_with_benign_context,
    trajectorize,
)


def test_alternating_roles_ends_opposite_target():
    roles = _alternating_roles_ending_before(Role.USER, n=3)
    assert len(roles) == 3
    assert roles[-1] is Role.CHAR  # immediately before a USER turn

    roles = _alternating_roles_ending_before(Role.CHAR, n=2)
    assert roles[-1] is Role.USER


def test_alternating_roles_empty():
    assert _alternating_roles_ending_before(Role.USER, n=0) == []


def test_build_benign_context_respects_target_role():
    rng = random.Random(0)
    bank = BenignContextBank.default()
    ctx = build_benign_context(n_turns=4, target_role=Role.USER, bank=bank, rng=rng)
    assert len(ctx) == 4
    # Before a USER target, last context turn should be CHAR
    assert ctx[-1].role is Role.CHAR
    # Alternation holds between consecutive turns
    for prev, curr in zip(ctx, ctx[1:], strict=False):
        assert prev.role != curr.role


def test_pad_with_benign_context_preserves_labels_and_adds_turns():
    rng = random.Random(1)
    ex = Example.from_text(
        "some harassment text",
        labels={Category.HARASSMENT: 0.9},
        source="t",
    )
    out = pad_with_benign_context(
        ex, BenignContextBank.default(), rng, n_turns=3, with_system_prompt_prob=0.0
    )
    assert len(out.conversation.turns) == 4  # 3 context + 1 target
    assert out.conversation.target.text == "some harassment text"
    assert out.labels == ex.labels
    assert "+benign_context" in out.source
    assert out.meta["augmentation"] == "benign_context"


def test_pad_with_benign_context_rejects_multi_turn_input():
    ex = Example(
        conversation=Conversation(
            turns=[
                Turn(role=Role.USER, text="hi"),
                Turn(role=Role.USER, text="target"),
            ]
        ),
        labels={},
    )
    with pytest.raises(ValueError):
        pad_with_benign_context(ex, BenignContextBank.default(), random.Random(0))


def test_pad_negative_rejects_positive_input():
    ex = Example.from_text("t", labels={Category.NSFW: 0.9})
    with pytest.raises(ValueError):
        pad_negative_with_edgy_context(ex, [], BenignContextBank.default(), random.Random(0))


def test_trajectorize_yields_at_least_some_outputs_and_preserves_multi_turn():
    negatives = [Example.from_text(f"clean text {i}", source="neg") for i in range(50)]
    positives = [
        Example.from_text(
            f"bad text {i}",
            labels={Category.HARASSMENT: 0.9},
            source="pos",
        )
        for i in range(50)
    ]
    # A pre-existing multi-turn example that should pass through unchanged.
    multi = Example(
        conversation=Conversation(
            turns=[
                Turn(role=Role.USER, text="hey"),
                Turn(role=Role.CHAR, text="hi"),
                Turn(role=Role.USER, text="target"),
            ]
        ),
        labels={Category.NSFW: 0.8},
        source="multi",
    )

    out = list(
        trajectorize(
            iter([*negatives, *positives, multi]),
            TrajectoryConfig(
                benign_pad_prob=1.0,
                edgy_neg_pad_prob=1.0,
                preserve_original_prob=1.0,
                seed=7,
            ),
        )
    )

    # With all probs at 1.0 we should emit both the original and the augmented
    # for every single-turn example, plus the multi-turn unchanged.
    assert len(out) >= len(negatives) + len(positives) + 1
    # The multi-turn Example must be passed through untouched
    multi_outputs = [e for e in out if e.source == "multi"]
    assert len(multi_outputs) == 1
    assert len(multi_outputs[0].conversation.turns) == 3


def test_benign_bank_from_examples_includes_real_data():
    examples = [Example.from_text(f"clean {i}", source="t") for i in range(20)]
    bank = BenignContextBank.from_examples(examples)
    # Bank should contain the defaults plus harvested texts
    assert any("clean 0" == t or "clean 1" == t for t in bank.user)
