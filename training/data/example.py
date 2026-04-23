from __future__ import annotations

from dataclasses import dataclass, field

from sage.conversation import Conversation, Role, Turn
from sage.schema import CATEGORIES, NUM_CATEGORIES, Category

ALL_CATEGORIES: frozenset[Category] = frozenset(CATEGORIES)


@dataclass(slots=True)
class Example:
    """Training example: a Conversation whose last turn is the classification target.

    ``observed`` lists the categories the source actually labeled. Unobserved
    positions are masked out of the training loss.
    """

    conversation: Conversation
    labels: dict[Category, float] = field(default_factory=dict)
    observed: frozenset[Category] = field(default_factory=lambda: ALL_CATEGORIES)
    source: str = ""
    meta: dict = field(default_factory=dict)

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
        return cls(
            conversation=Conversation.from_text(text, role=role),
            labels=labels or {},
            observed=observed if observed is not None else ALL_CATEGORIES,
            source=source,
            meta=meta or {},
        )

    @property
    def text(self) -> str:
        return self.conversation.target.text

    @property
    def target_turn(self) -> Turn:
        return self.conversation.target

    def to_vector(self) -> list[float]:
        return [self.labels.get(c, 0.0) for c in CATEGORIES]

    def observation_mask(self) -> list[float]:
        return [1.0 if c in self.observed else 0.0 for c in CATEGORIES]

    def is_negative(self) -> bool:
        return all(v < 0.5 for v in self.labels.values())

    @staticmethod
    def feature_dim() -> int:
        return NUM_CATEGORIES
