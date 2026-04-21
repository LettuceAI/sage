# Data pipeline

`sage` trains on an aggregated corpus drawn from six openly-licensed sources, with programmatic trajectory augmentation and an LLM-driven synthesis step for rare categories. This document describes each stage.

## Overview

```
six HF datasets
      │
      ▼
training.data.aggregate       ── produce train/val/test JSONL + stats
      │
      ▼
training.data.trajectorize_cli   ── wrap messages as conversations, add context
      │
      ▼
training.data.synthetic.cli      ── LLM-generated grooming / self-harm
      │                              trajectories → human review → merge
      ▼
training.train                   ── tokenize, train, checkpoint
```

## 1. Source aggregation

`training/data/loaders.py` defines one loader per source. Each yields `Example` objects tagged with `source` and a partial label map over the canonical 7-category schema.

| Source | Role in `sage` |
|---|---|
| Jigsaw Toxic Comments | Harassment, hate speech, violence (threats) |
| Civil Comments | Harassment, hate speech, violence, NSFW (sexual explicit) |
| Measuring Hate Speech | Hate speech (expert-annotated) |
| Salad-Data | NSFW, violence, self-harm, grooming, sexual minors |
| ProsocialDialog | Hard negatives (prosocial responses) |
| Anthropic HH-RLHF (red-team) | Diverse adversarial prompts across categories |

The aggregator CLI deduplicates across sources (SHA-1 fingerprint of normalized text) and emits three split files.

```bash
python -m training.data.aggregate \
    --out data/processed \
    --sources all \
    --limit-per-source 1000
```

Produces:

```
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/stats.json
```

Each line is:

```json
{
  "conversation": {"turns": [{"role": "user", "text": "..."}]},
  "labels": {"nsfw": 0.0, "harassment": 0.8, ...},
  "source": "civil_comments"
}
```

## 2. Trajectory augmentation

Source data is message-level. The trajectory augmenter wraps each `Example` as a one-turn conversation and probabilistically injects synthetic context:

- **Benign context padding.** Prepends unrelated user/character turns to a labeled target. Labels are preserved. Teaches the model to ignore irrelevant context.
- **Edgy-context hard negatives.** Prepends mildly toxic context to a clean-negative target. Label stays negative. Teaches the model that a toxic prior turn does not contaminate the current turn's classification.

```bash
python -m training.data.trajectorize_cli \
    --in  data/processed/train.jsonl \
    --out data/processed/train.traj.jsonl \
    --benign-pad-prob 0.6 \
    --edgy-neg-pad-prob 0.25 \
    --preserve-original-prob 0.4
```

Run per-split (train, val, test separately) so augmentation cannot leak the same target across splits.

## 3. Synthetic generation (rare categories)

Public datasets cover grooming and self-harm only sparsely. Trajectory-level detection in particular requires labeled escalation chains, which do not exist in bulk. The synthetic pipeline generates candidate trajectories via an LLM, holds them in a `pending/` directory, and merges only reviewer-approved examples into training data.

```bash
# 1. Generate
python -m training.data.synthetic.cli generate \
    --backend anthropic \
    --category grooming --polarity positive --n 50 \
    --out data/synthetic/pending/grooming_pos.jsonl

# 2. Human review — open each file and flip meta.review_status to
#    "approved" or "rejected" per line. Edit text where needed.

# 3. Merge approved rows into training-ready JSONL
python -m training.data.synthetic.cli merge \
    --in "data/synthetic/pending/*.jsonl" \
    --out data/processed/synthetic.jsonl

# 4. Concatenate with the main training set before trajectorization
cat data/processed/synthetic.jsonl >> data/processed/train.jsonl
```

### Policy

- **Sexual content involving minors** — the synthesis pipeline produces **only hard negatives** for this category (e.g. non-sexual references to young people). Positive signal for this category comes solely from the Salad-Data child-safety subset.
- **All synthetic output** is written with `meta.review_status = "pending"`. The merge CLI rejects anything not explicitly approved by a reviewer.

See [`training/data/synthetic/prompts.py`](../training/data/synthetic/prompts.py) for the full prompt text.

## 4. Statistics

The aggregator and trajectorizer both emit per-source and per-category counts in `stats.json`. Inspect these before training — rare positives (< a few thousand instances) are likely to underfit and may benefit from targeted synthesis.
