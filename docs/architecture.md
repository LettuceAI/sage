# Architecture

## Overview

| | |
|---|---|
| Base encoder | `jinaai/jina-embeddings-v2-base-en` (137M, 12 layers, 768 dim, ALiBi, 8192 native context) |
| Max input | 16 turns / 1024 tokens |
| Output | 7 sigmoid scores |
| Artifact | ~35 MB INT8 ONNX |

## Input format

Input is a `Conversation` — a non-empty list of `Turn` objects. The last turn is the classification target. Earlier turns are context.

```
[CLS] [role_1] text_1 [SEP] [role_2] text_2 [SEP] ... [CURRENT] target_text [SEP]
```

### Role tokens

| Token | Meaning |
|---|---|
| `[USER]` | End-user turn |
| `[CHAR]` | AI character / assistant turn |
| `[SYSTEM]` | System/persona prompt |
| `[CURRENT]` | Marker for the classification target (replaces the role token on the last turn) |

Added to the base tokenizer as 4 new special tokens. Embeddings initialized from the mean of semantic anchor words (`user`, `human`, `person` for `[USER]`; etc.) then fine-tuned.

### Length budget

| Parameter | Value |
|---|---|
| Max total tokens | 1024 |
| Max turns | 16 |

Truncation order:
1. Drop oldest context turns until the total fits.
2. If the target alone exceeds the budget, right-truncate.
3. `[CLS]` and the `[CURRENT]` marker are always kept.
4. A `[SYSTEM]` turn at index 0 is pinned unless keeping it forces target truncation.

At inference, messages exceeding the budget are chunked with overlap and max-aggregated (`sage.inference.Sage.check(..., chunk_long_messages=True)`).

## Pooling

Mean pooling over tokens in the `[CURRENT]` span (the `[CURRENT]` marker itself plus target-text tokens, excluding the terminating `[SEP]`). `[CLS]` and context tokens are masked out.

## Classifier head

```
pooled (768) → Dropout(0.2) → Linear(768 → 7) → sigmoid (at inference)
```

5,383 params.

## Training

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` with per-class `pos_weight`; focal γ=2 on `self_harm`, `grooming`, `sexual_minors` |
| Optimizer | AdamW, weight decay 0.01 |
| LR (layer-wise) | head 2e-4, top layers 8e-5, middle 3e-5, bottom 1e-5 |
| Schedule | 10% warmup, cosine decay |
| Precision | fp16 mixed |
| Max seq | 1024 (native, no interpolation needed) |

Stage 2 (continued fine-tune on synthetic data) uses 10× lower LRs.

### Partial-label supervision

Each source declares which categories it labels via a per-example `observed_mask`. BCE loss is multiplied by the mask so unobserved positions contribute zero gradient. This prevents partial-coverage sources from contaminating heads they never examined. See `DATA_LICENSES.md` for per-source coverage.

## Output

```json
{
  "flagged": true,
  "categories": {
    "nsfw":          {"score": 0.04, "flagged": false},
    "violence":      {"score": 0.12, "flagged": false},
    "harassment":    {"score": 0.08, "flagged": false},
    "hate_speech":   {"score": 0.03, "flagged": false},
    "self_harm":     {"score": 0.02, "flagged": false},
    "grooming":      {"score": 0.71, "flagged": true},
    "sexual_minors": {"score": 0.04, "flagged": false}
  }
}
```

`flagged` is true iff any category exceeds its threshold. Per-category thresholds are configurable via `Sage.set_threshold(category, value)`. Defaults live in `sage.schema.DEFAULT_THRESHOLDS`.
