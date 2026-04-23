"""Score a SAGE checkpoint on SAGE-Bench-Hard."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from sage.conversation import Conversation
from sage.model import build_model_and_tokenizer
from sage.schema import CATEGORIES, DEFAULT_THRESHOLDS, Category


def _score_row(model, tokenizer, text: str, device: torch.device) -> dict[Category, float]:
    enc = tokenizer.encode(Conversation.from_text(text))
    with torch.inference_mode():
        logits = model(
            torch.tensor([enc.input_ids], device=device),
            torch.tensor([enc.attention_mask], device=device),
            torch.tensor([enc.pooling_mask], device=device),
        )
        probs = torch.sigmoid(logits[0]).cpu().tolist()
    return {c: float(p) for c, p in zip(CATEGORIES, probs, strict=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SAGE against SAGE-Bench-Hard")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--bench",
        type=Path,
        default=Path(__file__).parent / "bench.jsonl",
    )
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument(
        "--thresholds",
        action="append",
        default=[],
        help="Override threshold for one category: --thresholds nsfw=0.7",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}  checkpoint={args.checkpoint}")
    model, tokenizer = build_model_and_tokenizer(max_length=args.max_length, dropout=0.0)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    thresholds = {c: DEFAULT_THRESHOLDS[c].threshold for c in CATEGORIES}
    for override in args.thresholds:
        k, v = override.split("=")
        thresholds[Category(k)] = float(v)
    print(f"[init] thresholds: {', '.join(f'{c.value}={t:.2f}' for c, t in thresholds.items())}")

    # Load bench
    rows: list[dict] = []
    with args.bench.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[init] loaded {len(rows)} benchmark rows")

    # Per-category confusion + failure-mode stats
    per_category_tp: dict[Category, int] = defaultdict(int)
    per_category_fp: dict[Category, int] = defaultdict(int)
    per_category_fn: dict[Category, int] = defaultdict(int)
    per_category_tn: dict[Category, int] = defaultdict(int)
    failure_mode_misses: dict[tuple[str, str], int] = defaultdict(int)
    failure_mode_totals: dict[tuple[str, str], int] = defaultdict(int)

    for i, row in enumerate(rows):
        if i % 100 == 0:
            print(f"  scoring {i}/{len(rows)}...")
        cat = Category(row["category"])
        expected_positive = row["expected"] == "positive"
        scores = _score_row(model, tokenizer, row["text"], device)
        predicted_positive = scores[cat] >= thresholds[cat]

        if expected_positive and predicted_positive:
            per_category_tp[cat] += 1
        elif expected_positive and not predicted_positive:
            per_category_fn[cat] += 1
            failure_mode_misses[(row["category"], row["failure_mode"])] += 1
        elif not expected_positive and predicted_positive:
            per_category_fp[cat] += 1
            failure_mode_misses[(row["category"], row["failure_mode"])] += 1
        else:
            per_category_tn[cat] += 1
        failure_mode_totals[(row["category"], row["failure_mode"])] += 1

    # Report per-category
    print("\n" + "=" * 70)
    print(
        f"{'Category':<16} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6}  {'Acc':>6}"
    )
    print("=" * 70)
    macro_f1s: list[float] = []
    for c in CATEGORIES:
        tp = per_category_tp[c]
        fp = per_category_fp[c]
        fn = per_category_fn[c]
        tn = per_category_tn[c]
        total = tp + fp + fn + tn
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        acc = (tp + tn) / max(1, total)
        if tp + fn > 0:
            macro_f1s.append(f1)
        print(
            f"{c.value:<16} {tp:>5} {fp:>5} {fn:>5} {tn:>5}  "
            f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}  {acc:>6.3f}"
        )

    print("-" * 70)
    print(f"{'Macro-F1':<16} {'':<22} {sum(macro_f1s) / max(1, len(macro_f1s)):>6.3f}")

    # Failure-mode breakdown — which patterns fail most
    print("\n" + "=" * 70)
    print("FAILURE MODES — top 20 worst-performing (by error rate)")
    print("=" * 70)
    print(f"{'category / failure_mode':<50} {'errs':>5} {'/':>1} {'tot':>5}  {'rate':>6}")
    print("-" * 70)
    mode_rates = []
    for key, total in failure_mode_totals.items():
        errs = failure_mode_misses.get(key, 0)
        rate = errs / total if total else 0
        if total >= 3:  # ignore one-off modes that have too few samples
            mode_rates.append((key, errs, total, rate))
    mode_rates.sort(key=lambda x: (-x[3], -x[2]))
    for (cat, mode), errs, total, rate in mode_rates[:20]:
        print(f"{cat + ' / ' + mode:<50} {errs:>5} / {total:>5}  {rate:>6.1%}")


if __name__ == "__main__":
    main()
