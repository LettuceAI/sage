# Training

This document covers the training loop, its default hyperparameters, and guidance for operating it on commodity hardware.

## Command

```bash
python -m training.train \
    --train data/processed/train.traj.jsonl \
    --val   data/processed/val.jsonl \
    --out   checkpoints/sage-v1 \
    --epochs 3 \
    --batch-size 32
```

See `python -m training.train --help` for the full argument list.

## Defaults

| Setting | Default | Rationale |
|---|---|---|
| Base model | `jinaai/jina-embeddings-v2-base-en` | 137M params, 12 layers, dim 768, native 8192 context via ALiBi |
| Max sequence length | 1024 | 16 turns × ~60 tokens average |
| Batch size | 32 | Fits an RTX 4060 8 GB at fp16 mixed precision |
| Gradient accumulation | 1 | Increase to achieve larger effective batches without VRAM pressure |
| Optimizer | AdamW, weight decay 0.01 | |
| Mixed precision | fp16 (CUDA AMP) | ~2× throughput on Ada/Ampere |
| Schedule | 10% warmup, cosine decay to 0 | |
| Loss | BCE with `pos_weight` per class; focal (γ = 2) for `self_harm`, `grooming`, `sexual_minors` | |

### Layer-wise learning-rate decay

```
classifier head    : 2e-4
encoder layers 10–11: 8e-5
encoder layers 6–9 : 3e-5
encoder layers 0–5 : 1e-5
```

Lower layers learn general linguistic features; upper layers and the head learn task-specific patterns. This schedule preserves the pretrained representations at the bottom while allowing the top to adapt freely.

## Metrics

The trainer evaluates on the validation split after every epoch and reports:

- Loss (averaged over the validation batches)
- Per-category precision, recall, and F1
- Macro-F1 (unweighted mean across the seven categories)

`best.pt` is saved whenever macro-F1 improves. `last.pt` is always the most recent epoch. `history.json` contains the full per-epoch breakdown.

Prefer macro-F1 over accuracy for model selection. Class imbalance makes accuracy uninformative — a model that predicts all-negatives would score 95%+ on most splits.

## Hardware guidance

| GPU | Batch 32 fp16 | Full run (500k × 2 epochs) |
|---|---|---|
| RTX 4060 8 GB | ~1.5–2 GB peak | ~3–4 hours |
| RTX 3090 24 GB | ~2 GB peak | ~2–3 hours |
| A100 40 GB | ~2 GB peak | ~1–1.5 hours |

For 8 GB cards: keep batch ≤ 32 and stay at 1024 tokens. If memory is tight, enable gradient checkpointing in `SageConfig` before training.

## Rare-class handling

Three categories — `self_harm`, `grooming`, `sexual_minors` — have very few positives in the raw corpus. The training defaults address this with:

1. **`pos_weight` per class.** Computed from training-set counts and clamped to 20. Upweights the loss on the rare positive side.
2. **Focal reshaping (γ = 2).** Down-weights easy examples so gradient flow stays concentrated on hard cases.
3. **Synthetic augmentation.** Reviewer-approved trajectories from the LLM-driven synthesis pipeline.

Inspect per-category F1 after each epoch. If a rare category plateaus below 0.5 F1, generate more synthetic data before increasing training time — more epochs on sparse data produces confident memorization, not generalization.

## Resuming

Resuming from a checkpoint is not automatic in v1. To continue training from `best.pt`, pass `--checkpoint path/to/best.pt` (planned) or manually load the state dict in the trainer. Resumption logic is tracked in the project roadmap.

## Export

Once training converges, export with:

```bash
python -m training.export_onnx \
    --checkpoint checkpoints/sage-v1/best.pt \
    --out-fp32   artifacts/sage-v1-fp32.onnx \
    --out-int8   artifacts/sage-v1-int8.onnx
```

The export script runs a parity check between PyTorch and ONNX outputs (and between ONNX fp32 and INT8) before finishing. See [`training/export_onnx.py`](../training/export_onnx.py).
