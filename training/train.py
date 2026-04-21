"""SAGE training loop.

Usage::

    python -m training.train \
        --train data/processed/train.jsonl \
        --val   data/processed/val.jsonl \
        --out   checkpoints/sage-v1 \
        --epochs 3 --batch-size 32

Defaults match ``docs/architecture.md`` §5 (layer-wise LR decay, fp16 mixed
precision, BCE + focal on rare classes, early stop on macro-F1).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sage.loss import SageLoss, compute_pos_weights
from sage.model import SageConfig, SageModel, build_model_and_tokenizer
from sage.schema import CATEGORIES
from sage.tokenizer import SageTokenizer
from training.dataset import SageDataset, collate


# ---------------------------------------------------------------------------
# Layer-wise LR decay — assigns smaller learning rates to earlier encoder layers
# ---------------------------------------------------------------------------
def build_param_groups(
    model: SageModel,
    *,
    lr_head: float = 2e-4,
    lr_top: float = 8e-5,
    lr_mid: float = 3e-5,
    lr_bot: float = 1e-5,
    weight_decay: float = 0.01,
    num_layers: int = 12,
) -> list[dict]:
    """Group parameters into 4 buckets for layer-wise LR decay.

    Jina v2 base has 12 transformer layers. We split:
        layers 0–5  → bottom LR
        layers 6–9  → middle LR
        layers 10–11 → top LR
        embeddings   → bottom LR
        everything else in the encoder → middle LR
        classifier + new special-token rows → head LR
    """
    head_params = list(model.classifier.parameters())

    enc_named = dict(model.encoder.named_parameters())

    bot, mid, top = [], [], []
    for name, p in enc_named.items():
        if not p.requires_grad:
            continue
        layer_idx = _extract_layer_index(name)
        if name.startswith("embeddings") or (layer_idx is not None and layer_idx < 6):
            bot.append(p)
        elif layer_idx is not None and layer_idx < 10:
            mid.append(p)
        elif layer_idx is not None and layer_idx < num_layers:
            top.append(p)
        else:
            # pooler, LN not inside a layer, etc.
            mid.append(p)

    return [
        {"params": bot, "lr": lr_bot, "weight_decay": weight_decay},
        {"params": mid, "lr": lr_mid, "weight_decay": weight_decay},
        {"params": top, "lr": lr_top, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]


def _extract_layer_index(name: str) -> int | None:
    """Pick layer index out of names like ``encoder.layer.4.attention...``."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p in ("layer", "layers") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Scheduler — warmup + cosine decay
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, *, num_training_steps: int, warmup_frac: float = 0.1):
    num_warmup = max(1, int(warmup_frac * num_training_steps))

    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return step / num_warmup
        progress = (step - num_warmup) / max(1, num_training_steps - num_warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(
    model: SageModel,
    loader: DataLoader,
    loss_fn: SageLoss,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["pooling_mask"])
        loss = loss_fn(logits, batch["labels"])
        total_loss += loss.item()
        n_batches += 1
        all_targets.append(batch["labels"].cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= threshold).astype(np.int32)
    targets_bin = (targets >= 0.5).astype(np.int32)

    per_class: dict[str, dict[str, float]] = {}
    for i, c in enumerate(CATEGORIES):
        per_class[c.value] = {
            "precision": float(
                precision_score(targets_bin[:, i], preds[:, i], zero_division=0)
            ),
            "recall": float(
                recall_score(targets_bin[:, i], preds[:, i], zero_division=0)
            ),
            "f1": float(f1_score(targets_bin[:, i], preds[:, i], zero_division=0)),
            "n_positive": int(targets_bin[:, i].sum()),
        }
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    return {
        "loss": total_loss / max(1, n_batches),
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def format_eval(results: dict) -> str:
    lines = [f"  loss={results['loss']:.4f}  macro_f1={results['macro_f1']:.4f}"]
    for name, m in results["per_class"].items():
        lines.append(
            f"    {name:<15} P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  n={m['n_positive']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[init] device={device}")

    model, tokenizer = build_model_and_tokenizer(
        base_model_name=args.base_model,
        max_length=args.max_length,
        dropout=args.dropout,
    )
    model.to(device)

    train_ds = SageDataset(args.train, tokenizer)
    val_ds = SageDataset(args.val, tokenizer)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    # Loss: pos_weight derived from training data
    label_counts = train_ds.label_counts()
    pos_weight = compute_pos_weights(label_counts, total=len(train_ds)).to(device)
    loss_fn = SageLoss(pos_weight=pos_weight).to(device)
    print(f"[loss] pos_weight={pos_weight.tolist()}")

    optimizer = AdamW(
        build_param_groups(
            model,
            lr_head=args.lr_head, lr_top=args.lr_top,
            lr_mid=args.lr_mid, lr_bot=args.lr_bot,
            weight_decay=args.weight_decay,
        ),
    )

    num_training_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = build_scheduler(optimizer, num_training_steps=num_training_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    history: list[dict] = []

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(batch["input_ids"], batch["attention_mask"], batch["pooling_mask"])
                loss = loss_fn(logits, batch["labels"]) / args.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            if (step + 1) % args.log_every == 0:
                avg = epoch_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch}  step {step + 1}/{len(train_loader)}  "
                      f"loss={avg:.4f}  lr={lr:.2e}")

        print(f"[epoch {epoch}] took {time.time() - t0:.1f}s  "
              f"avg_loss={epoch_loss / max(1, len(train_loader)):.4f}")
        eval_results = evaluate(model, val_loader, loss_fn, device)
        print(f"[epoch {epoch}] val:\n{format_eval(eval_results)}")
        history.append({"epoch": epoch, **eval_results})

        # Save last + best
        torch.save(
            {
                "model": model.state_dict(),
                "config": model.config.__dict__,
                "vocab_size": tokenizer.vocab_size,
                "epoch": epoch,
                "history": history,
            },
            out_dir / "last.pt",
        )
        if eval_results["macro_f1"] > best_macro_f1:
            best_macro_f1 = eval_results["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config.__dict__,
                    "vocab_size": tokenizer.vocab_size,
                    "epoch": epoch,
                    "macro_f1": best_macro_f1,
                },
                out_dir / "best.pt",
            )
            print(f"[epoch {epoch}] new best macro_f1={best_macro_f1:.4f} — saved")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[done] best macro_f1 = {best_macro_f1:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SAGE")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-model", default="jinaai/jina-embeddings-v2-base-en")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-head", type=float, default=2e-4)
    p.add_argument("--lr-top", type=float, default=8e-5)
    p.add_argument("--lr-mid", type=float, default=3e-5)
    p.add_argument("--lr-bot", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
