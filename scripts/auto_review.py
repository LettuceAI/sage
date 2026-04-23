"""Headless auto-review: apply heuristics, write approved rows to an output dir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import re
from collections import defaultdict


REFUSAL_PATTERNS = re.compile(
    "|".join(
        [
            r"\bi cannot help (you|with)\b",
            r"\bi can't help (you|with)\b",
            r"\bi cannot assist\b",
            r"\bi can't assist\b",
            r"\bas an ai\b",
            r"\bi'm not able to (help|assist|generate)\b",
            r"\bi am unable to\b",
            r"\bi will not (generate|produce|create)\b",
            r"\bi'm sorry,? but i (cannot|can't|won't|am)\b",
            r"\bi apologize,? but\b",
            r"\bi refuse to\b",
            r"\bagainst my guidelines\b",
            r"\bagainst openai\b",
        ]
    ),
    re.IGNORECASE,
)

SAFETY_GATE_PATTERNS = [
    "penis", "vagina", "blowjob", "masturbat", "undress",
    "naked pic", "nude pic", "sexting", "send nudes",
]


def _normalized_opening(text: str, length: int = 50) -> str:
    t = re.sub(r"\s+", " ", text.lower().strip())
    return t[:length]


def auto_review(rows: list[dict], max_per_cluster: int = 3) -> dict:
    pending_indices = [
        i for i, r in enumerate(rows)
        if (r.get("meta") or {}).get("review_status", "pending") == "pending"
    ]
    clusters: dict[str, list[int]] = defaultdict(list)
    for i in pending_indices:
        text = rows[i]["conversation"]["turns"][0]["text"]
        clusters[_normalized_opening(text)].append(i)
    duplicate_drop: set[int] = set()
    for _, idxs in clusters.items():
        for idx in idxs[max_per_cluster:]:
            duplicate_drop.add(idx)

    stats = {"total_pending_before": len(pending_indices), "approved": 0, "rejected": 0, "by_reason": defaultdict(int)}
    for i in pending_indices:
        row = rows[i]
        turns = row["conversation"]["turns"]
        reasons: list[str] = []
        if not (3 <= len(turns) <= 8):
            reasons.append("length_or_turns")
        elif not (20 <= len(turns[-1]["text"]) <= 500):
            reasons.append("length_or_turns")
        full_text = " ".join(t["text"] for t in turns)
        if REFUSAL_PATTERNS.search(full_text):
            reasons.append("refusal")
        if any(pat in full_text.lower() for pat in SAFETY_GATE_PATTERNS):
            reasons.append("safety_gate")
        if i in duplicate_drop:
            reasons.append("duplicate_opening")

        row.setdefault("meta", {})
        if reasons:
            row["meta"]["review_status"] = "rejected"
            row["meta"]["rejection_reasons"] = reasons
            row["meta"]["auto_reviewed"] = True
            stats["rejected"] += 1
            for reason in reasons:
                stats["by_reason"][reason] += 1
        else:
            row["meta"]["review_status"] = "approved"
            row["meta"]["auto_reviewed"] = True
            stats["approved"] += 1
    stats["by_reason"] = dict(stats["by_reason"])
    return stats


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(src: Path, out_approved_dir: Path, out_rejected_dir: Path | None) -> dict:
    rows = _read_jsonl(src)
    stats = auto_review(rows)
    approved = [r for r in rows if (r.get("meta") or {}).get("review_status") == "approved"]
    rejected = [r for r in rows if (r.get("meta") or {}).get("review_status") == "rejected"]

    out_approved_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_approved_dir / src.name, approved)
    if out_rejected_dir is not None:
        out_rejected_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_rejected_dir / src.name, rejected)

    return {
        "file": src.name,
        "in_rows": len(rows),
        "approved": len(approved),
        "rejected": len(rejected),
        "by_reason": stats.get("by_reason", {}),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run heuristic auto-review on synthesis JSONLs")
    p.add_argument("--in", dest="inp", type=Path, required=True, help="directory of *.jsonl to review")
    p.add_argument("--out", type=Path, required=True, help="output dir for approved files (flat)")
    p.add_argument("--rejected-out", type=Path, default=None, help="optional dir for rejected rows")
    args = p.parse_args()

    files = sorted(args.inp.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl files under {args.inp}")

    total_in = total_approved = total_rejected = 0
    reason_totals: dict[str, int] = {}

    print(f"{'file':<40} {'in':>7} {'ok':>7} {'drop':>7}")
    print("-" * 64)
    for src in files:
        r = process_file(src, args.out, args.rejected_out)
        print(f"{r['file']:<40} {r['in_rows']:>7} {r['approved']:>7} {r['rejected']:>7}")
        total_in += r["in_rows"]
        total_approved += r["approved"]
        total_rejected += r["rejected"]
        for k, v in r["by_reason"].items():
            reason_totals[k] = reason_totals.get(k, 0) + v

    print("-" * 64)
    keep_rate = total_approved / max(1, total_in)
    print(f"{'TOTAL':<40} {total_in:>7} {total_approved:>7} {total_rejected:>7}  ({keep_rate:.1%} kept)")
    if reason_totals:
        print("\nrejection reasons:")
        for reason, count in sorted(reason_totals.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<24} {count}")
    print(f"\napproved files written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
