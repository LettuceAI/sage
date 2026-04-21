from __future__ import annotations

from dataclasses import dataclass, field

from sage.conversation import Conversation, Role, Turn
from sage.schema import CATEGORIES, NUM_CATEGORIES, Category

ALL_CATEGORIES: frozenset[Category] = frozenset(CATEGORIES)


@dataclass(slots=True)
class Example:
    """Unified training example.

    A SAGE example is a ``Conversation`` (the last turn of which is the
    classification target) plus a set of category labels applied to that
    target turn, plus an explicit record of which categories the source
    actually observed.

    ``labels`` is a dict of ``Category -> float in [0, 1]``. Categories
    present in ``observed`` but absent from ``labels`` are treated as clean
    negatives (0). Categories not in ``observed`` are **unknown** — the
    trainer masks them out of the loss so the model never learns a fake
    "negative" for a category the source did not actually examine.

    Default ``observed = ALL_CATEGORIES`` is used only for sources that
    genuinely span every category (e.g. prosocial replies that cannot contain
    any of the seven harms). Partial-coverage sources (Measuring Hate Speech
    labels hate speech only, for example) must pass their explicit coverage.
    """

    conversation: Conversation
    labels: dict[Category, float] = field(default_factory=dict)
    observed: frozenset[Category] = field(default_factory=lambda: ALL_CATEGORIES)
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
        observed: frozenset[Category] | None = None,
        source: str = "",
        role: Role = Role.USER,
        meta: dict | None = None,
    ) -> Example:
        """Wrap a single message as a 1-turn example. Used by all message-level
        dataset loaders (Jigsaw, Civil Comments, etc.)."""
        return cls(
            conversation=Conversation.from_text(text, role=role),
            labels=labels or {},
            observed=observed if observed is not None else ALL_CATEGORIES,
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

    def observation_mask(self) -> list[float]:
        """1.0 for categories the source observed, 0.0 otherwise. The trainer
        multiplies BCE loss by this mask so unobserved categories contribute
        zero gradient."""
        return [1.0 if c in self.observed else 0.0 for c in CATEGORIES]

    def is_negative(self) -> bool:
        """True if every observed category is below 0.5."""
        return all(v < 0.5 for v in self.labels.values())

    @staticmethod
    def feature_dim() -> int:
        return NUM_CATEGORIES
