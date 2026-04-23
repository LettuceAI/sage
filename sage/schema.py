"""Seven-category multi-label schema."""

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
    threshold: float
    description: str


DEFAULT_THRESHOLDS: dict[Category, CategoryConfig] = {
    Category.NSFW: CategoryConfig(0.60, "Explicit sexual content between adults."),
    Category.VIOLENCE: CategoryConfig(0.55, "Graphic violence, gore, or threats."),
    Category.HARASSMENT: CategoryConfig(0.55, "Personal attacks, insults, targeted intimidation."),
    Category.HATE_SPEECH: CategoryConfig(0.50, "Attacks on protected classes."),
    Category.SELF_HARM: CategoryConfig(0.40, "Content promoting or describing self-harm."),
    Category.GROOMING: CategoryConfig(0.35, "Grooming patterns toward a minor."),
    Category.SEXUAL_MINORS: CategoryConfig(0.25, "Sexual content involving minors."),
}
