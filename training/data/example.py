from __future__ import annotations

from dataclasses import dataclass, field

from sage.conversation import Conversation, Role, Turn
from sage.schema import CATEGORIES, NUM_CATEGORIES, Category


@dataclass(slots=True)
class Example:
    """Unified training example.

    A SAGE example is a ``Conversation`` (the last turn of which is the
    classification target) plus a set of category labels applied to that
    target turn.

    ``labels`` is a dict of ``Category -> float in [0, 1]``. Missing categories
    are treated as 0 (not-flagged) by the trainer. Use ``to_vector`` for the
    dense 7-d target used by ``BCEWithLogitsLoss``.
    """

    conversation: Conversation
    labels: dict[Category, float] = field(default_factory=dict)
    source: str = ""
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_text(
        cls,
        text: str,
        labels: dict[Category, float] | None = None,
        source: str = "",
        role: Role = Role.USER,
        meta: dict | None = None,
    ) -> Example:
        """Wrap a single message as a 1-turn example. Used by all message-level
        dataset loaders (Jigsaw, Civil Comments, etc.)."""
        return cls(
            conversation=Conversation.from_text(text, role=role),
            labels=labels or {},
            source=source,
            meta=meta or {},
        )

    # ------------------------------------------------------------------
    # Backward-compatible accessors
    # ------------------------------------------------------------------
    @property
    def text(self) -> str:
        """Text of the target (last) turn. Useful for dedupe + inspection."""
        return self.conversation.target.text

    @property
    def target_turn(self) -> Turn:
        return self.conversation.target

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def to_vector(self) -> list[float]:
        return [self.labels.get(c, 0.0) for c in CATEGORIES]

    def is_negative(self) -> bool:
        """True if every category score is below 0.5."""
        return all(v < 0.5 for v in self.labels.values())

    @staticmethod
    def feature_dim() -> int:
        return NUM_CATEGORIES
