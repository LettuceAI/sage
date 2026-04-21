"""Canonical 7-label schema for SAGE.

All training data — regardless of source — is mapped into this schema.
Per-dataset mappings live in ``training/data/mappings.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    NSFW = "nsfw"
    VIOLENCE = "violence"
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    SELF_HARM = "self_harm"
    GROOMING = "grooming"
    SEXUAL_MINORS = "sexual_minors"


CATEGORIES: tuple[Category, ...] = tuple(Category)
NUM_CATEGORIES: int = len(CATEGORIES)
LABEL_TO_INDEX: dict[Category, int] = {c: i for i, c in enumerate(CATEGORIES)}
INDEX_TO_LABEL: dict[int, Category] = {i: c for i, c in enumerate(CATEGORIES)}


@dataclass(frozen=True)
class CategoryConfig:
    """Per-category inference config.

    ``threshold`` is the score above which ``flagged`` flips to True.
    Rare/high-cost categories use lower thresholds.
    """
    threshold: float
    description: str


DEFAULT_THRESHOLDS: dict[Category, CategoryConfig] = {
    Category.NSFW: CategoryConfig(
        threshold=0.60,
        description="Explicit sexual content between adults.",
    ),
    Category.VIOLENCE: CategoryConfig(
        threshold=0.55,
        description="Graphic violence, gore, or threats of physical harm.",
    ),
    Category.HARASSMENT: CategoryConfig(
        threshold=0.55,
        description="Personal attacks, insults, targeted intimidation.",
    ),
    Category.HATE_SPEECH: CategoryConfig(
        threshold=0.50,
        description="Attacks or slurs targeting protected classes.",
    ),
    Category.SELF_HARM: CategoryConfig(
        threshold=0.40,
        description="Content promoting, encouraging, or describing self-harm or suicide.",
    ),
    Category.GROOMING: CategoryConfig(
        threshold=0.35,
        description="Behavioral patterns indicative of grooming a minor.",
    ),
    Category.SEXUAL_MINORS: CategoryConfig(
        threshold=0.25,
        description="Sexual content depicting minors. Zero-tolerance category.",
    ),
}
