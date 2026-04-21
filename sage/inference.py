"""Python inference API for SAGE.

Loads an ONNX model (fp32 or INT8) + a tokenizer and exposes a friendly
``check()`` method that accepts either a string or a list of ``{role, text}``
dicts, matching the public API in the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from sage.conversation import Conversation, Role, Turn
from sage.schema import CATEGORIES, DEFAULT_THRESHOLDS, Category
from sage.tokenizer import SageTokenizer


MessageLike = Union[str, dict]
ConversationInput = Union[str, list[MessageLike]]


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

    def to_dict(self) -> dict:
        return {
            "flagged": self.flagged,
            "categories": {c.value: r.to_dict() for c, r in self.categories.items()},
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
            c: (thresholds[c] if thresholds and c in thresholds else DEFAULT_THRESHOLDS[c].threshold)
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
    ) -> "Sage":
        tokenizer = SageTokenizer(base_tokenizer_name=base_tokenizer, max_length=max_length)
        return cls(onnx_path, tokenizer, thresholds=thresholds, providers=providers)

    # ------------------------------------------------------------------
    def check(self, x: ConversationInput) -> ModerationResult:
        conv = _coerce_conversation(x)
        encoded = self.tokenizer.encode(conv)
        input_ids = np.asarray([encoded.input_ids], dtype=np.int64)
        attn = np.asarray([encoded.attention_mask], dtype=np.int64)
        pool = np.asarray([encoded.pooling_mask], dtype=np.int64)
        (logits,) = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attn, "pooling_mask": pool},
        )
        probs = _sigmoid(logits[0])  # (num_classes,)

        categories: dict[Category, CategoryResult] = {}
        any_flagged = False
        for i, c in enumerate(CATEGORIES):
            score = float(probs[i])
            flagged = score >= self._thresholds[c]
            any_flagged = any_flagged or flagged
            categories[c] = CategoryResult(score=score, flagged=flagged)
        return ModerationResult(flagged=any_flagged, categories=categories)

    def set_threshold(self, category: Category, threshold: float) -> None:
        self._thresholds[category] = threshold


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
