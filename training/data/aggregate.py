"""Aggregate all six source datasets into unified train/val/test JSONL files.

Usage::

    python -m training.data.aggregate --out data/processed --sources all

Writes:
    data/processed/train.jsonl
    data/processed/val.jsonl
    data/processed/test.jsonl
    data/processed/stats.json

Each JSONL line::

    {
        "conversation": {"turns": [{"role": "user", "text": "..."}]},
        "labels": {"nsfw": 0.0, ...},
        "source": "civil_comments"
    }

The ``conversation`` field is always a full ``Conversation`` object — for
message-level sources this is a 1-turn conversation; multi-turn trajectories
are emitted by the synthetic/trajectory pipelines (see
``training/data/trajectory.py``).

Deduplication: exact + near-duplicate (normalized whitespace, lowercased)
hashes across the entire pooled corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from sage.schema import CATEGORIES
from training.data.example import Example
from training.data.loaders import LOADERS

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def fingerprint(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def dedupe(examples: Iterable[Example]) -> Iterable[Example]:
    seen: set[str] = set()
    for ex in examples:
        if not ex.text or len(ex.text.strip()) < 3:
            continue
        fp = fingerprint(ex.text)
        if fp in seen:
            continue
        seen.add(fp)
        yield ex


def split_examples(
    examples: list[Example],
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[Example], list[Example], list[Example]]:
    rng = random.Random(seed)
    rng.shuffle(examples)
    n = len(examples)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = examples[:n_test]
    val = examples[n_test : n_test + n_val]
    train = examples[n_test + n_val :]
    return train, val, test


def serialize(ex: Example) -> dict:
    return {
        "conversation": ex.conversation.to_dict(),
        "labels": {c.value: ex.labels.get(c, 0.0) for c in CATEGORIES},
        "source": ex.source,
        **({"meta": ex.meta} if ex.meta else {}),
    }


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(serialize(ex), ensure_ascii=False) + "\n")


def compute_stats(examples: list[Example]) -> dict:
    per_source = Counter(ex.source for ex in examples)
    per_category_positive = Counter()
    for ex in examples:
        for c, v in ex.labels.items():
            if v >= 0.5:
                per_category_positive[c.value] += 1
    return {
        "total": len(examples),
        "per_source": dict(per_source),
        "per_category_positive": dict(per_category_positive),
        "pure_negatives": sum(1 for ex in examples if ex.is_negative()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SAGE training data")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["all"],
        help="Which datasets to include. 'all' or any subset of: " + ", ".join(LOADERS.keys()),
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=None,
        help="Cap examples per source (useful for smoke tests)",
    )
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = list(LOADERS.keys()) if args.sources == ["all"] else args.sources
    unknown = set(sources) - set(LOADERS.keys())
    if unknown:
        raise SystemExit(f"Unknown sources: {unknown}")

    pooled: list[Example] = []
    per_source_totals: dict[str, int] = {}

    for name in sources:
        print(f"[load] {name}")
        loader = LOADERS[name]
        count = 0
        for ex in loader():
            pooled.append(ex)
            count += 1
            if args.limit_per_source and count >= args.limit_per_source:
                break
        per_source_totals[name] = count
        print(f"[load] {name}: {count} examples")

    print(f"[dedupe] input={len(pooled)}")
    pooled = list(dedupe(pooled))
    print(f"[dedupe] output={len(pooled)}")

    train, val, test = split_examples(
        pooled, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(f"[split] train={len(train)} val={len(val)} test={len(test)}")

    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)
    write_jsonl(args.out / "test.jsonl", test)

    stats = {
        "loaded_per_source": per_source_totals,
        "after_dedupe": len(pooled),
        "splits": {
            "train": compute_stats(train),
            "val": compute_stats(val),
            "test": compute_stats(test),
        },
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[done] wrote splits + stats to {args.out}")


if __name__ == "__main__":
    main()
