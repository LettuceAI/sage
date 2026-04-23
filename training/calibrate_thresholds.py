"""Per-category threshold calibration on the val split.

Runs a single forward pass over val, caches probabilities and labels, then
sweeps thresholds per category to maximize F1 (computed only over observed
positions — categories a source never examined don't poison the metric).

Writes ``thresholds.json`` alongside the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from sage.model import build_model_and_tokenizer
from sage.schema import CATEGORIES, DEFAULT_THRESHOLDS, Category
from training.dataset import SageDataset, collate


@torch.inference_mode()
def collect_probs(model, loader, device: torch.device):
    model.eval()
    probs, targets, obs = [], [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["pooling_mask"])
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["labels"].cpu().numpy())
        if "observed_mask" in batch:
            obs.append(batch["observed_mask"].cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    targets = np.concatenate(targets, axis=0)
    obs = np.concatenate(obs, axis=0) if obs else np.ones_like(targets)
    return probs, (targets >= 0.5).astype(np.int32), obs.astype(bool)


def sweep_category(
    probs_col: np.ndarray,
    targets_col: np.ndarray,
    obs_col: np.ndarray,
    *,
    min_t: float = 0.05,
    max_t: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float, float, float]:
    """Return (best_threshold, precision, recall, f1) maximizing F1."""
    mask = obs_col
    if mask.sum() == 0 or targets_col[mask].sum() == 0:
        return 0.5, 0.0, 0.0, 0.0  # no positives to tune against
    p = probs_col[mask]
    t = targets_col[mask]
    best = (0.5, 0.0, 0.0, 0.0)
    for thr in np.arange(min_t, max_t + 1e-9, step):
        preds = (p >= thr).astype(np.int32)
        if preds.sum() == 0:
            continue
        prec = precision_score(t, preds, zero_division=0)
        rec = recall_score(t, preds, zero_division=0)
        f1 = f1_score(t, preds, zero_division=0)
        if f1 > best[3]:
            best = (float(thr), float(prec), float(rec), float(f1))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate per-category thresholds on val")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="Output thresholds.json path")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[init] device={device}  checkpoint={args.checkpoint}")

    model, tokenizer = build_model_and_tokenizer(max_length=args.max_length, dropout=0.0)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)

    val_ds = SageDataset(args.val, tokenizer)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[data] val={len(val_ds)}  batches={len(val_loader)}")

    print("[eval] computing probabilities on val...")
    probs, targets, obs = collect_probs(model, val_loader, device)
    print(f"[eval] probs shape={probs.shape}")

    # Per-category sweep
    print("\n" + "=" * 80)
    print(f"{'category':<16} {'old_t':>6} {'old_F1':>7}  →  "
          f"{'new_t':>6} {'P':>6} {'R':>6} {'F1':>7}  {'Δ':>6}")
    print("=" * 80)

    best_thresholds: dict[str, float] = {}
    old_f1s: list[float] = []
    new_f1s: list[float] = []
    for i, c in enumerate(CATEGORIES):
        default_t = DEFAULT_THRESHOLDS[c].threshold
        # Old metrics at default threshold
        mask = obs[:, i]
        if mask.sum() == 0 or targets[mask, i].sum() == 0:
            best_thresholds[c.value] = default_t
            print(f"{c.value:<16} {default_t:>6.2f} {'n/a':>7}  →  "
                  f"{'(no pos in val)':>36}")
            continue
        old_preds = (probs[mask, i] >= default_t).astype(np.int32)
        old_f1 = f1_score(targets[mask, i], old_preds, zero_division=0)

        new_t, new_p, new_r, new_f1 = sweep_category(probs[:, i], targets[:, i], obs[:, i])
        best_thresholds[c.value] = new_t
        old_f1s.append(old_f1)
        new_f1s.append(new_f1)
        delta = new_f1 - old_f1
        print(f"{c.value:<16} {default_t:>6.2f} {old_f1:>7.3f}  →  "
              f"{new_t:>6.2f} {new_p:>6.3f} {new_r:>6.3f} {new_f1:>7.3f}  "
              f"{'+' if delta >= 0 else ''}{delta:>6.3f}")

    print("-" * 80)
    if old_f1s and new_f1s:
        print(f"{'macro-F1':<16} {'':>6} {np.mean(old_f1s):>7.3f}  →  "
              f"{'':>13} {np.mean(new_f1s):>7.3f}  "
              f"+{np.mean(new_f1s) - np.mean(old_f1s):>5.3f}")

    out_path = args.out or (args.checkpoint.parent / "thresholds.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_thresholds, indent=2))
    print(f"\n[done] wrote thresholds to {out_path}")


if __name__ == "__main__":
    main()
