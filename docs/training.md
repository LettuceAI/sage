# Training

## Command

```bash
python -m training.train \
  --train data/processed/train.traj.jsonl \
  --val   data/processed/val.traj.jsonl \
  --out   checkpoints/sage-v1 \
  --epochs 3 \
  --batch-size 32
```

See `python -m training.train --help` for all flags.

## Defaults

| | |
|---|---|
| Base model | `jinaai/jina-embeddings-v2-base-en` |
| Max sequence length | 1024 |
| Batch size | 32 |
| Epochs | 3 |
| Precision | fp16 mixed (CUDA AMP) |
| Optimizer | AdamW, weight decay 0.01 |
| Schedule | 10% warmup, cosine decay |

### Layer-wise LR decay

```
classifier head        2e-4
encoder layers 10-11   8e-5
encoder layers 6-9     3e-5
encoder layers 0-5     1e-5
```

### Rare-class handling

`self_harm`, `grooming`, `sexual_minors` use focal loss (γ=2) with `pos_weight` derived from observed counts per class.

## Metrics

Per epoch, validated on the `val` split:
- Loss (focal-BCE, masked by observation)
- Per-category precision, recall, F1
- Macro-F1 over observed categories

`best.pt` is saved when macro-F1 improves. `last.pt` is the most recent epoch. `history.json` is the full log.

## Hardware

| GPU | Full 3-epoch run (≈186k examples, batch 32 / fp16) |
|---|---|
| RTX 4060 8 GB | ~5 hrs (batch 4 × grad_accum 8, seq 256) |
| L4 24 GB | ~1.5-2 hrs (batch 16 × grad_accum 2, seq 1024) |
| A100 40 GB | ~45-60 min (batch 32, seq 1024) |

Use `--resume-from path/to/best.pt` for stage-2 fine-tuning on top of an existing checkpoint. Lower LRs by ~10× when resuming.

## Export

```bash
python -m training.export_onnx \
  --checkpoint checkpoints/sage-v1/best.pt \
  --out-fp32   artifacts/sage-v1-fp32.onnx \
  --out-int8   artifacts/sage-v1-int8.onnx
```

Runs a PyTorch-vs-ONNX parity check at fp32 and INT8 before finishing.
