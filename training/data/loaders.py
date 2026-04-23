"""Per-source loaders. Each yields Examples with labels + observed categories."""

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


# civil_comments — google/civil_comments, CC0. harassment/violence/hate/nsfw.
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


# measuring_hate_speech — ucberkeley-dlab/measuring-hate-speech, CC-BY-4.0.
# Labels hate_speech only; middle-range scores dropped as noise.
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


# salad_data — OpenSafetyLab/Salad-Data, Apache 2.0. Broad taxonomy.
SALAD_OBSERVED: frozenset[Category] = ALL_CATEGORIES

SALAD_TAXONOMY_SUBSTR: tuple[tuple[str, Category], ...] = (
    # Sexual content involving minors / grooming — route narrow, specific tags
    # to GROOMING since Salad's "Child Abuse" is broad behavioral harm. We do
    # not currently map anything to SEXUAL_MINORS from Salad; the category
    # taxonomy doesn't carve it out specifically enough. Dedicated CSAM signal
    # comes from NVIDIA Aegis.
    ("child abuse", Category.GROOMING),
    # Self-harm
    ("self-harm", Category.SELF_HARM),
    ("self harm", Category.SELF_HARM),
    ("suicide", Category.SELF_HARM),
    # NSFW (adult sexual content)
    ("erotic", Category.NSFW),
    ("sexual activity", Category.NSFW),
    ("adult content", Category.NSFW),
    ("sexual offense", Category.NSFW),
    # Violence
    ("violent crime", Category.VIOLENCE),
    ("violent content", Category.VIOLENCE),
    ("weapon", Category.VIOLENCE),
    ("biological and chemical harm", Category.VIOLENCE),
    # Hate speech / stereotyping / discrimination
    ("hate speech", Category.HATE_SPEECH),
    ("stereotyping", Category.HATE_SPEECH),
    # Harassment-style bullying/insult/targeted meanness
    ("harass", Category.HARASSMENT),
    ("bully", Category.HARASSMENT),
    ("insult", Category.HARASSMENT),
    ("enjoying someone else's pain", Category.HARASSMENT),
)


def _salad_labels_for_tag(tag: str) -> Category | None:
    lower = tag.lower()
    for needle, category in SALAD_TAXONOMY_SUBSTR:
        if needle in lower:
            return category
    return None


def load_salad_data(subset: str = "base_set") -> Iterator[Example]:
    from datasets import load_dataset

    ds = load_dataset("OpenSafetyLab/Salad-Data", subset, split="train")
    for row in ds:
        text = row.get("question") or row.get("baseq") or row.get("augq") or ""
        if not text:
            continue
        labels: dict[Category, float] = {}
        for key in ("1-category", "2-category", "3-category"):
            tag = row.get(key)
            if not tag:
                continue
            mapped = _salad_labels_for_tag(tag)
            if mapped is not None:
                labels[mapped] = 1.0
        if not labels:
            continue
        yield Example.from_text(
            text,
            labels=labels,
            observed=SALAD_OBSERVED,
            source="salad_data",
        )


# prosocial_dialog — allenai/prosocial-dialog, CC-BY-4.0. All-category negatives.
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


# hh_rlhf_redteam — Anthropic/hh-rlhf red-team subset, MIT. Tag-based labels.
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


# aegis — nvidia/Aegis-AI-Content-Safety-Dataset-1.0, CC-BY-4.0.
# Broad taxonomy including a dedicated Sexual Minor label.
AEGIS_CATEGORY_MAP: dict[str, Category] = {
    "hate/identity hate": Category.HATE_SPEECH,
    "hate": Category.HATE_SPEECH,
    "identity hate": Category.HATE_SPEECH,
    "sexual": Category.NSFW,
    "violence": Category.VIOLENCE,
    "threat": Category.VIOLENCE,
    "suicide and self harm": Category.SELF_HARM,
    "suicide": Category.SELF_HARM,
    "self harm": Category.SELF_HARM,
    "sexual minor": Category.SEXUAL_MINORS,
    "harassment": Category.HARASSMENT,
    # Aegis categories we deliberately don't map (out of SAGE scope):
    # profanity, guns/illegal weapons, controlled substances,
    # criminal planning, pii/privacy, other.
}


def _aegis_labels_from_row(row: dict) -> tuple[dict[Category, float], bool] | None:
    """Return (labels, is_explicit_safe) or None if the row is unlabelled.

    ``is_explicit_safe`` indicates the row was marked "Safe" by the annotator —
    which gives us permission to claim negative across all seven categories.
    """
    # Different Aegis revisions expose slightly different column names. Try a
    # sequence of likely fields to find the annotator's category labels.
    candidates = []
    for key in ("labels_0", "labels_1", "labels_2", "violated_categories", "categories"):
        value = row.get(key)
        if value:
            candidates.append(value)
    if not candidates:
        return None

    # Normalize candidates to a flat list of lowercase strings.
    flat: list[str] = []
    for value in candidates:
        if isinstance(value, str):
            flat.extend(v.strip().lower() for v in value.split(","))
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, str):
                    flat.append(v.strip().lower())

    if not flat:
        return None

    is_safe = any(v in ("safe",) for v in flat)
    labels: dict[Category, float] = {}
    for tag in flat:
        for key, category in AEGIS_CATEGORY_MAP.items():
            if key in tag:
                labels[category] = 1.0
                break
    return labels, is_safe


def load_aegis(split: str = "train") -> Iterator[Example]:
    """Emit Examples from NVIDIA's Aegis Content Safety dataset."""
    ds = _load_hf("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", split=split)
    for row in ds:
        text = row.get("text") or row.get("prompt") or ""
        if not text:
            continue
        # Aegis labels both prompts and responses. Skip the response rows for
        # v1 — they are LLM outputs and we want SAGE to classify user-side
        # text first. Support for response-side classification can come later.
        text_type = (row.get("text_type") or "").lower()
        if text_type and "response" in text_type:
            continue

        parsed = _aegis_labels_from_row(row)
        if parsed is None:
            continue
        labels, is_safe = parsed

        if is_safe and not labels:
            # Explicit "Safe" — reliable clean negative across all categories.
            yield Example.from_text(
                text,
                labels={},
                observed=ALL_CATEGORIES,
                source="aegis",
            )
        elif labels:
            yield Example.from_text(
                text,
                labels=labels,
                observed=frozenset(labels.keys()),
                source="aegis",
            )


# wildchat — allenai/WildChat-1M, ODC-BY. Per-turn OpenAI Moderation labels.
WILDCHAT_OPENAI_MAP: dict[str, Category] = {
    "harassment": Category.HARASSMENT,
    "harassment/threatening": Category.HARASSMENT,
    "hate": Category.HATE_SPEECH,
    "hate/threatening": Category.HATE_SPEECH,
    "self-harm": Category.SELF_HARM,
    "self-harm/intent": Category.SELF_HARM,
    "self-harm/instructions": Category.SELF_HARM,
    "sexual": Category.NSFW,
    "sexual/minors": Category.SEXUAL_MINORS,
    "violence": Category.VIOLENCE,
    "violence/graphic": Category.VIOLENCE,
}


def _wildchat_labels_from_scores(
    category_scores: dict | None,
    *,
    threshold: float = 0.5,
) -> dict[Category, float]:
    if not isinstance(category_scores, dict):
        return {}
    labels: dict[Category, float] = {}
    for key, score in category_scores.items():
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if score_f < threshold:
            continue
        mapped = WILDCHAT_OPENAI_MAP.get(key)
        if mapped is None:
            continue
        labels[mapped] = max(labels.get(mapped, 0.0), score_f)
    return labels


def load_wildchat(split: str = "train") -> Iterator[Example]:
    """Emit one Example per USER turn across WildChat conversations.

    Uses the OpenAI Moderation ``category_scores`` when present; otherwise
    falls back to the Boolean ``categories`` dict. Rows without any
    moderation payload are treated as full-coverage negatives — WildChat's
    cleaned release already filtered the most toxic conversations upstream.
    """
    ds = _load_hf("allenai/WildChat-1M", split=split)
    for row in ds:
        conversation = row.get("conversation") or []
        moderations = row.get("openai_moderation") or []
        if not isinstance(conversation, list):
            continue
        for turn_idx, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                continue
            if (turn.get("role") or "").lower() != "user":
                continue
            text = turn.get("content") or ""
            if not text:
                continue
            mod = moderations[turn_idx] if turn_idx < len(moderations) else None
            scores = (mod or {}).get("category_scores") if isinstance(mod, dict) else None
            labels = _wildchat_labels_from_scores(scores)
            yield Example.from_text(
                text,
                labels=labels,
                observed=ALL_CATEGORIES,
                source="wildchat",
                role=Role.USER,
                meta={"turn": turn_idx},
            )


# Registry
LOADERS = {
    "civil_comments": load_civil_comments,
    "measuring_hate_speech": load_measuring_hate_speech,
    "salad_data": load_salad_data,
    "prosocial_dialog": load_prosocial_dialog,
    "hh_rlhf_redteam": load_hh_rlhf_redteam,
    "aegis": load_aegis,
    "wildchat": load_wildchat,
}
