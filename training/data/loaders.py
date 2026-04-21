"""Dataset loaders. Each loader yields ``Example`` objects with labels mapped
into SAGE's canonical 7-category schema.

Load lazily — none of these import ``datasets`` at module level so unit tests
can import ``loaders`` without the optional training dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from sage.conversation import Role
from sage.schema import Category
from training.data.example import Example

if TYPE_CHECKING:
    from datasets import Dataset


def _load_hf(name: str, split: str, **kwargs) -> Dataset:
    from datasets import load_dataset

    return load_dataset(name, split=split, **kwargs)


# ---------------------------------------------------------------------------
# 1. Jigsaw Toxic Comment Classification
#    Columns: comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
#    License: CC0 + CC-BY-SA-3.0
# ---------------------------------------------------------------------------
def load_jigsaw(split: str = "train") -> Iterator[Example]:
    ds = _load_hf("google/jigsaw_toxicity_pred", split=split, trust_remote_code=True)
    for row in ds:
        labels: dict[Category, float] = {}
        # Harassment = any generic toxicity/insult signal
        harass = max(row["toxic"], row["severe_toxic"], row["insult"])
        if harass:
            labels[Category.HARASSMENT] = float(harass)
        if row["threat"]:
            labels[Category.VIOLENCE] = float(row["threat"])
        if row["identity_hate"]:
            labels[Category.HATE_SPEECH] = float(row["identity_hate"])
        # Deliberately NOT mapping `obscene` → NSFW; obscene is mostly profanity,
        # not explicit sexual content. Would introduce label noise.
        yield Example.from_text(
            row["comment_text"],
            labels=labels,
            source="jigsaw",
        )


# ---------------------------------------------------------------------------
# 2. Civil Comments
#    Columns: text, toxicity, severe_toxicity, obscene, threat, insult,
#             identity_attack, sexual_explicit (all float ∈ [0, 1])
#    License: CC0
# ---------------------------------------------------------------------------
def load_civil_comments(split: str = "train", min_score: float = 0.3) -> Iterator[Example]:
    ds = _load_hf("google/civil_comments", split=split)
    for row in ds:
        labels: dict[Category, float] = {}
        if row["sexual_explicit"] >= min_score:
            labels[Category.NSFW] = float(row["sexual_explicit"])
        if row["threat"] >= min_score:
            labels[Category.VIOLENCE] = float(row["threat"])
        harass = max(row["toxicity"], row["severe_toxicity"], row["insult"])
        if harass >= min_score:
            labels[Category.HARASSMENT] = float(harass)
        if row["identity_attack"] >= min_score:
            labels[Category.HATE_SPEECH] = float(row["identity_attack"])
        yield Example.from_text(
            row["text"],
            labels=labels,
            source="civil_comments",
        )


# ---------------------------------------------------------------------------
# 3. Measuring Hate Speech (UC Berkeley)
#    Columns: text, hate_speech_score (continuous, ≈[-8, +4])
#    License: CC-BY-4.0
# ---------------------------------------------------------------------------
def load_measuring_hate_speech(split: str = "train") -> Iterator[Example]:
    ds = _load_hf("ucberkeley-dlab/measuring-hate-speech", split=split)
    for row in ds:
        # Normalize: score > 0.5 → positive hate speech; scale sigmoid-style
        score = float(row["hate_speech_score"])
        if score <= -1:
            intensity = 0.0
        elif score >= 2:
            intensity = 1.0
        else:
            intensity = (score + 1) / 3  # linear in [-1, 2] → [0, 1]
        labels: dict[Category, float] = {}
        if intensity > 0.3:
            labels[Category.HATE_SPEECH] = intensity
        yield Example.from_text(
            row["text"],
            labels=labels,
            source="measuring_hate_speech",
        )


# ---------------------------------------------------------------------------
# 4. Salad-Data (OpenSafetyLab)
#    Hierarchical taxonomy (1/2/3 level). Prompt text in `question` or `baseq`.
#    License: Apache 2.0
#
#    Mapping is the most complex of the six — refine on first real run by
#    inspecting the taxonomy values actually present in the data.
# ---------------------------------------------------------------------------
SALAD_TAXONOMY_MAP: dict[str, Category] = {
    # level-3 labels → SAGE category. Expand after first run.
    "Sexual Activity and Erotic Content": Category.NSFW,
    "Adult Content": Category.NSFW,
    "Violence and Physical Harm": Category.VIOLENCE,
    "Graphic Violence": Category.VIOLENCE,
    "Hate Speech and Discrimination": Category.HATE_SPEECH,
    "Harassment and Bullying": Category.HARASSMENT,
    "Self-harm and Suicide": Category.SELF_HARM,
    "Child Sexual Abuse Material": Category.SEXUAL_MINORS,
    "Child Abuse and Exploitation": Category.GROOMING,
}


def load_salad_data(subset: str = "base_set") -> Iterator[Example]:
    ds = _load_hf("OpenSafetyLab/Salad-Data", subset, split="train")
    for row in ds:
        text = row.get("question") or row.get("baseq") or row.get("augq") or ""
        if not text:
            continue
        labels: dict[Category, float] = {}
        # Salad labels are strings at 1/2/3-category depth; map any that match.
        for key in ("1-category", "2-category", "3-category"):
            tag = row.get(key)
            if tag and tag in SALAD_TAXONOMY_MAP:
                labels[SALAD_TAXONOMY_MAP[tag]] = 1.0
        yield Example.from_text(
            text,
            labels=labels,
            source="salad_data",
        )


# ---------------------------------------------------------------------------
# 5. ProsocialDialog — HARD NEGATIVES
#    safety_label ∈ {__casual__, __possibly_needs_caution__,
#                    __probably_needs_caution__, __needs_caution__,
#                    __needs_intervention__}
#    License: CC-BY-4.0
#
#    We use `response` (the prosocial reply) as a clean hard negative — these
#    are texts about sensitive topics that are themselves not violations.
# ---------------------------------------------------------------------------
def load_prosocial_dialog(split: str = "train") -> Iterator[Example]:
    ds = _load_hf("allenai/prosocial-dialog", split=split)
    for row in ds:
        # The response is prosocial → all-zero labels (hard negative).
        # Tagged with role=CHAR since prosocial replies come from the assistant side.
        yield Example.from_text(
            row["response"],
            labels={},  # clean negative
            source="prosocial_dialog",
            role=Role.CHAR,
            meta={"safety_label": row.get("safety_label", "")},
        )


# ---------------------------------------------------------------------------
# 6. Anthropic HH-RLHF — red-team subset only
#    License: MIT
#
#    The `tags` field often contains harm categories (e.g. "violence",
#    "racism"). Use the transcript as a positive example of the tagged harm.
# ---------------------------------------------------------------------------
HH_TAG_MAP: dict[str, Category] = {
    "violence": Category.VIOLENCE,
    "weapons": Category.VIOLENCE,
    "racism": Category.HATE_SPEECH,
    "discrimination": Category.HATE_SPEECH,
    "bullying": Category.HARASSMENT,
    "harassment": Category.HARASSMENT,
    "sexual": Category.NSFW,
    "self-harm": Category.SELF_HARM,
    "suicide": Category.SELF_HARM,
    "child": Category.SEXUAL_MINORS,
    "minor": Category.SEXUAL_MINORS,
}


def load_hh_rlhf_redteam() -> Iterator[Example]:
    ds = _load_hf("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
    for row in ds:
        tags = row.get("tags") or []
        labels: dict[Category, float] = {}
        for tag in tags:
            tag_lower = str(tag).lower()
            for key, category in HH_TAG_MAP.items():
                if key in tag_lower:
                    labels[category] = 1.0
        yield Example.from_text(
            row["transcript"],
            labels=labels,
            source="hh_rlhf_redteam",
            meta={"harmlessness_score": row.get("min_harmlessness_score_transcript")},
        )


# ---------------------------------------------------------------------------
# Registry — used by aggregate.py
# ---------------------------------------------------------------------------
LOADERS = {
    "jigsaw": load_jigsaw,
    "civil_comments": load_civil_comments,
    "measuring_hate_speech": load_measuring_hate_speech,
    "salad_data": load_salad_data,
    "prosocial_dialog": load_prosocial_dialog,
    "hh_rlhf_redteam": load_hh_rlhf_redteam,
}
