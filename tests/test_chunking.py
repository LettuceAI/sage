"""Tests for inference-time chunking of oversize single messages.

These exercise the merge + budget logic directly without requiring an ONNX
session — we construct a partial Sage instance by subclassing, and build
fake ModerationResult partials to drive ``_merge_max``.
"""

from dataclasses import dataclass

import pytest

from sage.inference import CategoryResult, ModerationResult, Sage
from sage.schema import CATEGORIES, DEFAULT_THRESHOLDS, Category


class _SageForTests(Sage):
    """Skip __init__ — lets us exercise helpers without needing an ONNX model."""

    def __init__(self, thresholds=None):  # noqa: D401
        self._thresholds = {c: DEFAULT_THRESHOLDS[c].threshold for c in CATEGORIES}
        if thresholds:
            self._thresholds.update(thresholds)
        # tokenizer and _session intentionally unset


def _result(**scores) -> ModerationResult:
    cats: dict[Category, CategoryResult] = {}
    for c in CATEGORIES:
        s = scores.get(c.value, 0.0)
        cats[c] = CategoryResult(score=s, flagged=s >= DEFAULT_THRESHOLDS[c].threshold)
    return ModerationResult(
        flagged=any(r.flagged for r in cats.values()),
        categories=cats,
        n_chunks=1,
    )


# ---------------------------------------------------------------------------
# _merge_max
# ---------------------------------------------------------------------------
def test_merge_max_picks_highest_per_category():
    s = _SageForTests()
    partials = [
        _result(nsfw=0.1, violence=0.3),
        _result(nsfw=0.9, violence=0.2),
        _result(nsfw=0.4, violence=0.8),
    ]
    merged = s._merge_max(partials)
    assert merged.categories[Category.NSFW].score == pytest.approx(0.9)
    assert merged.categories[Category.VIOLENCE].score == pytest.approx(0.8)
    assert merged.n_chunks == 3


def test_merge_max_recomputes_flags_at_merge():
    """If any chunk crosses the threshold, the merged result should flag."""
    s = _SageForTests()
    # NSFW default threshold is 0.60. Neither partial exceeds on its own for
    # violence (threshold 0.55), but one partial hits NSFW above threshold.
    partials = [
        _result(nsfw=0.5, violence=0.5),
        _result(nsfw=0.7, violence=0.3),
    ]
    merged = s._merge_max(partials)
    assert merged.categories[Category.NSFW].flagged is True
    assert merged.categories[Category.VIOLENCE].flagged is False
    assert merged.flagged is True


def test_merge_max_respects_custom_thresholds():
    """Raising the threshold should demote a previously-flagged result."""
    s = _SageForTests(thresholds={Category.NSFW: 0.95})
    partials = [_result(nsfw=0.7)]
    merged = s._merge_max(partials)
    assert merged.categories[Category.NSFW].flagged is False
    assert merged.flagged is False


def test_merge_max_rejects_empty():
    s = _SageForTests()
    with pytest.raises(AssertionError):
        s._merge_max([])


# ---------------------------------------------------------------------------
# _target_budget
# ---------------------------------------------------------------------------
@dataclass
class _StubTokenizer:
    """Minimal stand-in for SageTokenizer exposing just what _target_budget uses."""

    max_length: int
    # nested stub representing the HF tokenizer
    tokenizer: object

    def encode(self, *a, **k):
        raise NotImplementedError


class _StubHFTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Crude word-tokenize: one ID per whitespace-separated token
        assert add_special_tokens is False, "budget helper must not add specials"
        return [0] * len(text.split())


def _stub_sage(max_length: int = 64) -> _SageForTests:
    s = _SageForTests()
    s.tokenizer = _StubTokenizer(max_length=max_length, tokenizer=_StubHFTokenizer())
    return s


def test_target_budget_without_context():
    from sage.conversation import Conversation

    s = _stub_sage(max_length=50)
    conv = Conversation.from_text("hello world")
    # Budget = max_length - 1 (CLS) - 2 (CURRENT + SEP) = 47
    assert s._target_budget(conv) == 47


def test_target_budget_shrinks_with_each_context_turn():
    from sage.conversation import Conversation, Role, Turn

    s = _stub_sage(max_length=50)
    conv = Conversation(
        turns=[
            Turn(role=Role.USER, text="one two three"),  # 3 tokens + 2 overhead
            Turn(role=Role.CHAR, text="four"),  # 1 token + 2 overhead
            Turn(role=Role.USER, text="target"),
        ]
    )
    # Budget = 50 - 1 (CLS) - 2 (target wrap) - (2+3) (ctx1) - (2+1) (ctx2) = 39
    assert s._target_budget(conv) == 39


def test_target_budget_floor_at_one():
    """Even with huge context, budget never returns <1."""
    from sage.conversation import Conversation, Role, Turn

    s = _stub_sage(max_length=10)
    conv = Conversation(
        turns=[
            Turn(role=Role.USER, text="one " * 200),  # massive context
            Turn(role=Role.USER, text="target"),
        ]
    )
    assert s._target_budget(conv) >= 1
