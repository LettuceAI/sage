"""Python inference API for SAGE.

Loads an ONNX model (fp32 or INT8) + a tokenizer and exposes a friendly
``check()`` method that accepts either a string or a list of ``{role, text}``
dicts, matching the public API in the README.

Single-message content that exceeds the model's ``max_length`` is automatically
chunked: the target message is split into overlapping token windows, each
window is classified with the full conversation context, and the per-category
scores are aggregated with an element-wise max. Chunking is transparent to the
caller and can be disabled via ``chunk_long_messages=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sage.conversation import Conversation, Role, Turn
from sage.schema import CATEGORIES, DEFAULT_THRESHOLDS, Category
from sage.tokenizer import SageTokenizer

MessageLike = str | dict
ConversationInput = str | list[MessageLike]


@dataclass
class CategoryResult:
    score: float
    flagged: bool

    def to_dict(self) -> dict:
        return {"score": self.score, "flagged": self.flagged}


@dataclass
class ModerationResult:
    flagged: bool
    categories: dict[Category, CategoryResult]
    n_chunks: int = 1  # number of chunks used — 1 for non-chunked calls

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "categories": {c.value: r.to_dict() for c, r in self.categories.items()},
            "n_chunks": self.n_chunks,
        }


class Sage:
    """Inference-time wrapper around an ONNX SAGE model."""

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer: SageTokenizer,
        thresholds: dict[Category, float] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.tokenizer = tokenizer
        self._thresholds: dict[Category, float] = {
            c: (
                thresholds[c] if thresholds and c in thresholds else DEFAULT_THRESHOLDS[c].threshold
            )
            for c in CATEGORIES
        }
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=providers or ["CPUExecutionProvider"],
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        *,
        base_tokenizer: str = "jinaai/jina-embeddings-v2-base-en",
        max_length: int = 1024,
        thresholds: dict[Category, float] | None = None,
        providers: list[str] | None = None,
    ) -> Sage:
        tokenizer = SageTokenizer(base_tokenizer_name=base_tokenizer, max_length=max_length)
        return cls(onnx_path, tokenizer, thresholds=thresholds, providers=providers)

    # ------------------------------------------------------------------
    def check(
        self,
        x: ConversationInput,
        *,
        chunk_long_messages: bool = True,
        chunk_overlap: int = 64,
    ) -> ModerationResult:
        """Classify a message or conversation.

        When the target message's token length exceeds the available budget
        (``max_length`` minus context / special-token overhead), the target is
        split into overlapping windows, each classified with the full context,
        and the per-category scores are max-aggregated. This preserves full
        content coverage on long inputs without retraining at a larger
        sequence length.

        Args:
            x: a string, a list of ``{role, text}`` dicts, or a list of
                ``Turn`` objects.
            chunk_long_messages: set ``False`` to disable chunking — oversized
                targets will then be right-truncated by the tokenizer's normal
                policy (fast but may miss violations in the dropped tail).
            chunk_overlap: tokens of overlap between adjacent chunks. Prevents
                missing violations that straddle chunk boundaries.
        """
        conv = _coerce_conversation(x)

        if not chunk_long_messages:
            return self._classify(conv, n_chunks=1)

        target_ids = self.tokenizer.tokenizer.encode(
            conv.target.text, add_special_tokens=False
        )
        target_budget = self._target_budget(conv)

        if len(target_ids) <= target_budget:
            return self._classify(conv, n_chunks=1)

        # Split the target into overlapping windows, re-build a Conversation
        # per chunk, and classify each.
        stride = max(1, target_budget - chunk_overlap)
        partials: list[ModerationResult] = []
        start = 0
        while start < len(target_ids):
            end = min(start + target_budget, len(target_ids))
            chunk_ids = target_ids[start:end]
            chunk_text = self.tokenizer.tokenizer.decode(
                chunk_ids, skip_special_tokens=True
            )
            partial_conv = Conversation(
                turns=[*conv.context, Turn(role=conv.target.role, text=chunk_text)]
            )
            partials.append(self._classify(partial_conv, n_chunks=1))
            if end >= len(target_ids):
                break
            start += stride

        return self._merge_max(partials)

    def set_threshold(self, category: Category, threshold: float) -> None:
        self._thresholds[category] = threshold

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _classify(self, conv: Conversation, *, n_chunks: int) -> ModerationResult:
        encoded = self.tokenizer.encode(conv)
        input_ids = np.asarray([encoded.input_ids], dtype=np.int64)
        attn = np.asarray([encoded.attention_mask], dtype=np.int64)
        pool = np.asarray([encoded.pooling_mask], dtype=np.int64)
        (logits,) = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attn, "pooling_mask": pool},
        )
        probs = _sigmoid(logits[0])

        categories: dict[Category, CategoryResult] = {}
        any_flagged = False
        for i, c in enumerate(CATEGORIES):
            score = float(probs[i])
            flagged = score >= self._thresholds[c]
            any_flagged = any_flagged or flagged
            categories[c] = CategoryResult(score=score, flagged=flagged)
        return ModerationResult(flagged=any_flagged, categories=categories, n_chunks=n_chunks)

    def _merge_max(self, partials: list[ModerationResult]) -> ModerationResult:
        """Element-wise max across per-chunk scores. Re-applies thresholds."""
        assert partials, "no partials to merge"
        categories: dict[Category, CategoryResult] = {}
        any_flagged = False
        for c in CATEGORIES:
            max_score = max(p.categories[c].score for p in partials)
            flagged = max_score >= self._thresholds[c]
            any_flagged = any_flagged or flagged
            categories[c] = CategoryResult(score=max_score, flagged=flagged)
        return ModerationResult(
            flagged=any_flagged,
            categories=categories,
            n_chunks=len(partials),
        )

    def _target_budget(self, conv: Conversation) -> int:
        """Max tokens the target message can occupy without forcing truncation.

        Mirrors the budget math in ``sage/tokenizer.py``: reserve 1 token for
        ``[CLS]``, ``2 + len(ctx)`` tokens per context turn for role + SEP, and
        2 for the target's own ``[CURRENT]`` + trailing ``[SEP]``.
        """
        budget = self.tokenizer.max_length - 1  # [CLS]
        budget -= 2  # [CURRENT] + [SEP] for the target
        for turn in conv.context:
            ctx_ids = self.tokenizer.tokenizer.encode(turn.text, add_special_tokens=False)
            budget -= 2 + len(ctx_ids)
        return max(1, budget)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_conversation(x: ConversationInput) -> Conversation:
    if isinstance(x, str):
        return Conversation.from_text(x)
    if isinstance(x, list):
        turns: list[Turn] = []
        for item in x:
            if isinstance(item, Turn):
                turns.append(item)
            elif isinstance(item, dict):
                turns.append(Turn(role=Role(item["role"]), text=str(item["text"])))
            else:
                raise TypeError(f"expected Turn or dict, got {type(item).__name__}")
        return Conversation(turns=turns)
    raise TypeError(f"expected str or list, got {type(x).__name__}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
