"""Orchestrate synthetic trajectory generation.

The builder turns a ``Generator`` and a category request into validated
``Example`` objects, each tagged as synthetic and pending human review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from sage.conversation import Conversation, Role, Turn
from sage.schema import Category
from training.data.example import Example
from training.data.synthetic.generator import Generator, json_loads_lenient
from training.data.synthetic.prompts import CATEGORY_PROMPTS, SYSTEM_PROMPT


@dataclass
class SyntheticBatch:
    examples: list[Example] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def extend(self, other: "SyntheticBatch") -> None:
        self.examples.extend(other.examples)
        self.errors.extend(other.errors)


class SyntheticBuilder:
    """Generate synthetic training trajectories via an LLM ``Generator``.

    Outputs are **always** tagged with ``meta["review_status"] = "pending"``.
    They MUST NOT be merged into training data until a human reviewer flips
    that to ``"approved"`` — the merge CLI enforces this.
    """

    def __init__(self, generator: Generator, system_prompt: str | None = None) -> None:
        self.generator = generator
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def build(
        self,
        category: str,
        polarity: str,
        n: int,
        *,
        batch_size: int = 10,
        max_tokens: int = 4096,
        temperature: float = 0.9,
    ) -> SyntheticBatch:
        """Generate ``n`` examples of ``(category, polarity)``.

        ``polarity`` ∈ ``{"positive", "negative"}``. Positive means the label
        should be set for ``category``; negative means the label is 0.

        Raises ``ValueError`` for the policy-forbidden case of
        ``(category="sexual_minors", polarity="positive")``.
        """
        if category not in CATEGORY_PROMPTS:
            raise ValueError(f"unknown category for synthesis: {category!r}")
        pos_prompt, neg_prompt = CATEGORY_PROMPTS[category]
        if polarity == "positive":
            prompt_template = pos_prompt
        elif polarity == "negative":
            prompt_template = neg_prompt
        else:
            raise ValueError(f"polarity must be 'positive' or 'negative', got {polarity!r}")

        if prompt_template is None:
            raise ValueError(
                f"policy forbids synthetic positives for category={category}"
            )

        batch = SyntheticBatch()
        remaining = n
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            try:
                raw = self.generator.generate(
                    system=self.system_prompt,
                    user=prompt_template.format(n=this_batch),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                rows = json_loads_lenient(raw)
            except (ValueError, RuntimeError) as e:
                batch.errors.append(f"batch failed: {e}")
                remaining -= this_batch
                continue

            if not isinstance(rows, list):
                batch.errors.append("generator did not return a JSON array")
                remaining -= this_batch
                continue

            for row in rows:
                try:
                    ex = self._row_to_example(row, category=category, polarity=polarity)
                    batch.examples.append(ex)
                except (KeyError, ValueError, TypeError) as e:
                    batch.errors.append(f"skipped row: {e}")
            remaining -= this_batch

        return batch

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def grooming_positives(self, n: int, **kw) -> SyntheticBatch:
        return self.build("grooming", "positive", n, **kw)

    def grooming_negatives(self, n: int, **kw) -> SyntheticBatch:
        return self.build("grooming", "negative", n, **kw)

    def selfharm_positives(self, n: int, **kw) -> SyntheticBatch:
        return self.build("self_harm", "positive", n, **kw)

    def selfharm_negatives(self, n: int, **kw) -> SyntheticBatch:
        return self.build("self_harm", "negative", n, **kw)

    def minors_negatives(self, n: int, **kw) -> SyntheticBatch:
        return self.build("sexual_minors", "negative", n, **kw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_example(row: dict, *, category: str, polarity: str) -> Example:
        turns_raw = row["turns"]
        if not isinstance(turns_raw, list) or not turns_raw:
            raise ValueError("turns must be a non-empty list")
        turns: list[Turn] = []
        for t in turns_raw:
            role = Role(t["role"])
            text = str(t["text"]).strip()
            if not text:
                raise ValueError("empty turn text")
            turns.append(Turn(role=role, text=text))

        labels: dict[Category, float] = {}
        if polarity == "positive":
            labels[Category(category)] = 1.0

        return Example(
            conversation=Conversation(turns=turns),
            labels=labels,
            source=f"synthetic_{category}_{polarity}",
            meta={
                "review_status": "pending",
                "generator_notes": row.get("notes", ""),
                "category": category,
                "polarity": polarity,
            },
        )


def iter_examples_for_review(batch: SyntheticBatch) -> Iterable[Example]:
    """Filter helper — used by the review CLI."""
    for ex in batch.examples:
        if ex.meta.get("review_status") == "pending":
            yield ex
