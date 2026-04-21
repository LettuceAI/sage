"""CLI for synthetic trajectory generation and human review.

Two-step workflow:

1. **Generate**::

       python -m training.data.synthetic.cli generate \
           --category grooming --polarity positive --n 50 \
           --out data/synthetic/pending/grooming_pos.jsonl

   Produces a JSONL file with ``meta.review_status = "pending"`` on every row.

2. **Human review**: a human edits the file, flipping ``review_status`` to
   ``"approved"`` or ``"rejected"`` per row. They may also tweak the text or
   labels to correct generator mistakes.

3. **Merge**::

       python -m training.data.synthetic.cli merge \
           --in data/synthetic/pending/*.jsonl \
           --out data/processed/synthetic.jsonl

   Picks up only ``review_status == "approved"`` rows and emits them in the
   same JSONL shape as the aggregator output, ready to concatenate with the
   main training set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from sage.conversation import Conversation
from sage.schema import CATEGORIES, Category
from training.data.example import Example
from training.data.synthetic.builder import SyntheticBuilder
from training.data.synthetic.generator import (
    AnthropicGenerator,
    MockGenerator,
    OllamaGenerator,
)


def _make_generator(backend: str, model: str | None):
    if backend == "anthropic":
        kwargs = {"model": model} if model else {}
        return AnthropicGenerator(**kwargs)
    if backend == "ollama":
        kwargs = {"model": model} if model else {}
        return OllamaGenerator(**kwargs)
    if backend == "mock":
        # Mock requires canned responses via env var (JSON list of strings)
        responses = json.loads(os.environ.get("SAGE_MOCK_RESPONSES", "[]"))
        return MockGenerator(responses=responses)
    raise SystemExit(f"unknown backend: {backend}")


def _write_jsonl(path: Path, examples) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(_serialize(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


def _serialize(ex: Example) -> dict:
    out = {
        "conversation": ex.conversation.to_dict(),
        "labels": {c.value: ex.labels.get(c, 0.0) for c in CATEGORIES},
        "source": ex.source,
    }
    if ex.meta:
        out["meta"] = ex.meta
    return out


def _deserialize(row: dict) -> Example:
    conv = Conversation.from_dict(row["conversation"])
    labels: dict[Category, float] = {
        Category(k): float(v) for k, v in (row.get("labels") or {}).items() if float(v) > 0
    }
    return Example(
        conversation=conv,
        labels=labels,
        source=row.get("source", ""),
        meta=row.get("meta", {}),
    )


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def cmd_generate(args: argparse.Namespace) -> None:
    generator = _make_generator(args.backend, args.model)
    builder = SyntheticBuilder(generator)
    batch = builder.build(
        category=args.category,
        polarity=args.polarity,
        n=args.n,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )
    n = _write_jsonl(args.out, batch.examples)
    print(f"[gen] wrote {n} pending examples to {args.out}")
    if batch.errors:
        print(f"[gen] {len(batch.errors)} errors during generation:")
        for e in batch.errors[:5]:
            print(f"  - {e}")


def cmd_merge(args: argparse.Namespace) -> None:
    approved: list[dict] = []
    counts: dict[str, int] = {"approved": 0, "rejected": 0, "pending": 0, "other": 0}
    for pattern in args.inp:
        for path_str in sorted(glob.glob(pattern)):
            path = Path(path_str)
            for row in _read_jsonl(path):
                status = (row.get("meta") or {}).get("review_status", "pending")
                counts[status] = counts.get(status, 0) + 1
                if status == "approved":
                    approved.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in approved:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[merge] approved={counts['approved']}  rejected={counts['rejected']}  "
          f"pending={counts['pending']}")
    print(f"[merge] wrote {len(approved)} approved examples to {args.out}")


def cmd_stats(args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for pattern in args.inp:
        for path_str in sorted(glob.glob(pattern)):
            for row in _read_jsonl(Path(path_str)):
                status = (row.get("meta") or {}).get("review_status", "pending")
                key = f"{row.get('source', '')}:{status}"
                counts[key] = counts.get(key, 0) + 1
    for k in sorted(counts):
        print(f"  {k:<60} {counts[k]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE synthetic trajectory pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate pending synthetic examples")
    g.add_argument("--backend", choices=["anthropic", "ollama", "mock"], default="anthropic")
    g.add_argument("--model", default=None, help="Override backend model id")
    g.add_argument("--category", required=True, choices=["grooming", "self_harm", "sexual_minors"])
    g.add_argument("--polarity", required=True, choices=["positive", "negative"])
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--batch-size", type=int, default=10)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--out", type=Path, required=True)
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("merge", help="Merge approved reviews into training-ready JSONL")
    m.add_argument("--in", dest="inp", nargs="+", required=True, help="glob patterns")
    m.add_argument("--out", type=Path, required=True)
    m.set_defaults(func=cmd_merge)

    s = sub.add_parser("stats", help="Show approved/pending/rejected counts")
    s.add_argument("--in", dest="inp", nargs="+", required=True, help="glob patterns")
    s.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
