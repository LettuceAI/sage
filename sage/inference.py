"""Inference API for an exported ONNX SAGE model."""

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
    n_chunks: int = 1

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "categories": {c.value: r.to_dict() for c, r in self.categories.items()},
            "n_chunks": self.n_chunks,
        }


class Sage:
    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer: SageTokenizer,
        thresholds: dict[Category, float] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.tokenizer = tokenizer
        self._thresholds = {
            c: (
                thresholds[c] if thresholds and c in thresholds else DEFAULT_THRESHOLDS[c].threshold
            )
            for c in CATEGORIES
        }
        self._session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )

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

    def check(
        self,
        x: ConversationInput,
        *,
        chunk_long_messages: bool = True,
        chunk_overlap: int = 64,
    ) -> ModerationResult:
        """Classify a message or conversation. Oversize targets are chunked and max-aggregated."""
        conv = _coerce_conversation(x)
        if not chunk_long_messages:
            return self._classify(conv, n_chunks=1)

        target_ids = self.tokenizer.tokenizer.encode(conv.target.text, add_special_tokens=False)
        budget = self._target_budget(conv)
        if len(target_ids) <= budget:
            return self._classify(conv, n_chunks=1)

        stride = max(1, budget - chunk_overlap)
        partials: list[ModerationResult] = []
        start = 0
        while start < len(target_ids):
            end = min(start + budget, len(target_ids))
            chunk_text = self.tokenizer.tokenizer.decode(
                target_ids[start:end], skip_special_tokens=True
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

    def _classify(self, conv: Conversation, *, n_chunks: int) -> ModerationResult:
        enc = self.tokenizer.encode(conv)
        (logits,) = self._session.run(
            None,
            {
                "input_ids": np.asarray([enc.input_ids], dtype=np.int64),
                "attention_mask": np.asarray([enc.attention_mask], dtype=np.int64),
                "pooling_mask": np.asarray([enc.pooling_mask], dtype=np.int64),
            },
        )
        probs = _sigmoid(logits[0])
        categories: dict[Category, CategoryResult] = {}
        any_flagged = False
        for i, c in enumerate(CATEGORIES):
            s = float(probs[i])
            f = s >= self._thresholds[c]
            any_flagged = any_flagged or f
            categories[c] = CategoryResult(score=s, flagged=f)
        return ModerationResult(flagged=any_flagged, categories=categories, n_chunks=n_chunks)

    def _merge_max(self, partials: list[ModerationResult]) -> ModerationResult:
        assert partials
        categories: dict[Category, CategoryResult] = {}
        any_flagged = False
        for c in CATEGORIES:
            s = max(p.categories[c].score for p in partials)
            f = s >= self._thresholds[c]
            any_flagged = any_flagged or f
            categories[c] = CategoryResult(score=s, flagged=f)
        return ModerationResult(flagged=any_flagged, categories=categories, n_chunks=len(partials))

    def _target_budget(self, conv: Conversation) -> int:
        """Tokens the target can occupy without forcing truncation. Mirrors SageTokenizer."""
        budget = self.tokenizer.max_length - 1 - 2  # [CLS] + target ([CURRENT], [SEP])
        for turn in conv.context:
            ctx_ids = self.tokenizer.tokenizer.encode(turn.text, add_special_tokens=False)
            budget -= 2 + len(ctx_ids)
        return max(1, budget)


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
