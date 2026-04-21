"""Dataset loaders. Each loader yields ``Example`` objects with labels mapped
into SAGE's canonical 7-category schema.

Each source declares a module-level ``OBSERVED`` constant — the set of
categories that source actually labels. The trainer masks loss on unobserved
categories so sources that cover only a subset of the taxonomy do not
contaminate the other heads with spurious negatives.

Load lazily — none of these import ``datasets`` at module level so unit tests
can import ``loaders`` without the optional training dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from sage.conversation import Role
from sage.schema import Category
from training.data.example import ALL_CATEGORIES, Example

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
JIGSAW_OBSERVED: frozenset[Category] = frozenset(
    {Category.HARASSMENT, Category.VIOLENCE, Category.HATE_SPEECH}
)


def load_jigsaw(split: str = "train") -> Iterator[Example]:
    ds = _load_hf("google/jigsaw_toxicity_pred", split=split, trust_remote_code=True)
    for row in ds:
        labels: dict[Category, float] = {}
        harass = max(row["toxic"], row["severe_toxic"], row["insult"])
        if harass:
            labels[Category.HARASSMENT] = float(harass)
        if row["threat"]:
            labels[Category.VIOLENCE] = float(row["threat"])
        if row["identity_hate"]:
            labels[Category.HATE_SPEECH] = float(row["identity_hate"])
        # `obscene` is mostly profanity, not explicit sexual content; deliberately
        # not mapped to NSFW. Jigsaw doesn't observe NSFW at all.
        yield Example.from_text(
            row["comment_text"],
            labels=labels,
            observed=JIGSAW_OBSERVED,
            source="jigsaw",
        )


# ---------------------------------------------------------------------------
# 2. Civil Comments
#    Columns: text, toxicity, severe_toxicity, obscene, threat, insult,
#             identity_attack, sexual_explicit (all float ∈ [0, 1])
#    License: CC0
# ---------------------------------------------------------------------------
CIVIL_COMMENTS_OBSERVED: frozenset[Category] = frozenset(
    {Category.HARASSMENT, Category.VIOLENCE, Category.HATE_SPEECH, Category.NSFW}
)


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
            observed=CIVIL_COMMENTS_OBSERVED,
            source="civil_comments",
        )


# ---------------------------------------------------------------------------
# 3. Measuring Hate Speech (UC Berkeley)
#    Columns: text, hate_speech_score (continuous, ≈[-8, +4])
#    License: CC-BY-4.0
#
#    Annotators scored hate speech only. The other six SAGE categories are
#    unobserved — never inferred from a low ``hate_speech_score``.
#
#    Middle-range scores are dropped: crowd-annotated boundary cases produce
#    more label noise than signal for a classifier head.
# ---------------------------------------------------------------------------
MHS_OBSERVED: frozenset[Category] = frozenset({Category.HATE_SPEECH})


def load_measuring_hate_speech(
    split: str = "train",
    *,
    drop_middle_low: float = -0.5,
    drop_middle_high: float = 0.5,
) -> Iterator[Example]:
    ds = _load_hf("ucberkeley-dlab/measuring-hate-speech", split=split)
    for row in ds:
        score = float(row["hate_speech_score"])
        if drop_middle_low < score < drop_middle_high:
            continue
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
            observed=MHS_OBSERVED,
            source="measuring_hate_speech",
        )


# ---------------------------------------------------------------------------
# 4. Salad-Data (OpenSafetyLab)
#    License: Apache 2.0
#
#    Salad is a broad safety dataset. Rows whose 1/2/3-category tag matches
#    our taxonomy emit a positive label. Rows whose tags do not match any of
#    our categories are skipped rather than emitted as silent all-negatives.
# ---------------------------------------------------------------------------
SALAD_OBSERVED: frozenset[Category] = ALL_CATEGORIES

SALAD_TAXONOMY_MAP: dict[str, Category] = {
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
        for key in ("1-category", "2-category", "3-category"):
            tag = row.get(key)
            if tag and tag in SALAD_TAXONOMY_MAP:
                labels[SALAD_TAXONOMY_MAP[tag]] = 1.0
        if not labels:
            # No mapped tag — skip rather than emit a fake all-negative.
            continue
        yield Example.from_text(
            text,
            labels=labels,
            observed=SALAD_OBSERVED,
            source="salad_data",
        )


# ---------------------------------------------------------------------------
# 5. ProsocialDialog — HARD NEGATIVES across all categories
#    License: CC-BY-4.0
#
#    Prosocial replies are, by construction, free of all seven harms — full
#    observation across the taxonomy is appropriate here.
# ---------------------------------------------------------------------------
PROSOCIAL_OBSERVED: frozenset[Category] = ALL_CATEGORIES


def load_prosocial_dialog(split: str = "train") -> Iterator[Example]:
    ds = _load_hf("allenai/prosocial-dialog", split=split)
    for row in ds:
        yield Example.from_text(
            row["response"],
            labels={},
            observed=PROSOCIAL_OBSERVED,
            source="prosocial_dialog",
            role=Role.CHAR,
            meta={"safety_label": row.get("safety_label", "")},
        )


# ---------------------------------------------------------------------------
# 6. Anthropic HH-RLHF — red-team subset
#    License: MIT
#
#    Observed is per-row, derived from the ``tags`` field. Transcripts without
#    any mapped harm tag are skipped (we cannot assert what kind of harm they
#    are, and they are not reliable negatives since the dataset is harm-seeded).
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
        if not labels:
            continue
        yield Example.from_text(
            row["transcript"],
            labels=labels,
            observed=frozenset(labels.keys()),
            source="hh_rlhf_redteam",
            meta={"harmlessness_score": row.get("min_harmlessness_score_transcript")},
        )


# ---------------------------------------------------------------------------
# 7. WildGuardMix (AI2) — purpose-built safety classifier training data
#    License: ODC-BY (commercial-permissive; AI2 Responsible Use Guidelines apply)
#
#    Every row is annotator-verified across harm types. ``prompt_harm_label``
#    gives a global harm verdict; ``subcategory`` narrows the specific type.
#    Unharmful rows are the one source we trust to provide reliable negatives
#    across all seven categories simultaneously: if annotators marked a prompt
#    "unharmful" they examined every kind of harm.
# ---------------------------------------------------------------------------
WILDGUARD_OBSERVED_UNHARMFUL: frozenset[Category] = ALL_CATEGORIES

# Ordered substring matches — first hit wins. ``sexual_minors`` patterns are
# checked before the generic ``sexual`` pattern so CSAM-adjacent rows route to
# the right head.
WILDGUARD_SUBCAT_MAP: tuple[tuple[str, Category], ...] = (
    ("child sexual", Category.SEXUAL_MINORS),
    ("csam", Category.SEXUAL_MINORS),
    ("sexual content involving minors", Category.SEXUAL_MINORS),
    ("minors", Category.SEXUAL_MINORS),
    ("grooming", Category.GROOMING),
    ("self-harm", Category.SELF_HARM),
    ("self harm", Category.SELF_HARM),
    ("suicide", Category.SELF_HARM),
    ("sexual", Category.NSFW),
    ("explicit", Category.NSFW),
    ("violence", Category.VIOLENCE),
    ("physical harm", Category.VIOLENCE),
    ("weapon", Category.VIOLENCE),
    ("harassment", Category.HARASSMENT),
    ("bullying", Category.HARASSMENT),
    ("abuse", Category.HARASSMENT),
    ("hate", Category.HATE_SPEECH),
    ("discriminat", Category.HATE_SPEECH),
    ("stereotype", Category.HATE_SPEECH),
)


def _wildguard_map_subcategory(sub: str | None) -> Category | None:
    if not sub:
        return None
    s = sub.lower()
    for needle, category in WILDGUARD_SUBCAT_MAP:
        if needle in s:
            return category
    return None


def load_wildguardmix(split: str = "train") -> Iterator[Example]:
    """Emit one Example per WildGuardMix prompt.

    Policy:
    - unharmful prompt → clean negative across all 7 categories
    - harmful prompt, subcategory maps → positive for that single category,
      observed = {that category} (can't assert clean on the other six)
    - harmful prompt, subcategory unmapped → skip
    - response-only rows (no prompt_harm_label) → skip for v1
    """
    subset = "wildguardtrain" if split == "train" else "wildguardtest"
    from datasets import load_dataset

    ds = load_dataset("allenai/wildguardmix", subset, split="train")
    for row in ds:
        prompt = row.get("prompt") or ""
        if not prompt:
            continue
        harm_label = (row.get("prompt_harm_label") or "").lower()
        subcategory = row.get("subcategory")

        if harm_label == "unharmful":
            yield Example.from_text(
                prompt,
                labels={},
                observed=WILDGUARD_OBSERVED_UNHARMFUL,
                source="wildguardmix",
                meta={"adversarial": bool(row.get("adversarial", False))},
            )
        elif harm_label == "harmful":
            mapped = _wildguard_map_subcategory(subcategory)
            if mapped is None:
                continue
            yield Example.from_text(
                prompt,
                labels={mapped: 1.0},
                observed=frozenset({mapped}),
                source="wildguardmix",
                meta={
                    "adversarial": bool(row.get("adversarial", False)),
                    "subcategory": subcategory,
                },
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
    "wildguardmix": load_wildguardmix,
}
