"""Augment an aggregated JSONL with trajectory context. Run per split."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from sage.conversation import Conversation
from sage.schema import CATEGORIES, Category
from training.data.example import ALL_CATEGORIES, Example
from training.data.trajectory import TrajectoryConfig, trajectorize


def _deserialize(row: dict) -> Example:
    conv = Conversation.from_dict(row["conversation"])
    labels: dict[Category, float] = {
        Category(k): float(v) for k, v in (row.get("labels") or {}).items() if v > 0
    }
    # Back-compat: older JSONL lines lack "observed"; treat those as fully
    # observed (matches the pre-masking default).
    observed_raw = row.get("observed")
    if observed_raw is None:
        observed = ALL_CATEGORIES
    else:
        observed = frozenset(Category(v) for v in observed_raw)
    return Example(
        conversation=conv,
        labels=labels,
        observed=observed,
        source=row.get("source", ""),
        meta=row.get("meta", {}),
    )


def _serialize(ex: Example) -> dict:
    out = {
        "conversation": ex.conversation.to_dict(),
        "labels": {c.value: ex.labels.get(c, 0.0) for c in CATEGORIES},
        "observed": sorted(c.value for c in ex.observed),
        "source": ex.source,
    }
    if ex.meta:
        out["meta"] = ex.meta
    return out


def read_examples(path: Path) -> Iterator[Example]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield _deserialize(json.loads(line))


def write_examples(path: Path, examples: Iterator[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(_serialize(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectorize an aggregated SAGE JSONL")
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benign-pad-prob", type=float, default=0.6)
    parser.add_argument("--edgy-neg-pad-prob", type=float, default=0.25)
    parser.add_argument("--preserve-original-prob", type=float, default=0.4)
    args = parser.parse_args()

    cfg = TrajectoryConfig(
        benign_pad_prob=args.benign_pad_prob,
        edgy_neg_pad_prob=args.edgy_neg_pad_prob,
        preserve_original_prob=args.preserve_original_prob,
        seed=args.seed,
    )

    examples = list(read_examples(args.inp))
    print(f"[traj] read {len(examples)} from {args.inp}")
    n_out = write_examples(args.out, trajectorize(iter(examples), cfg))
    print(f"[traj] wrote {n_out} to {args.out}")


if __name__ == "__main__":
    main()
