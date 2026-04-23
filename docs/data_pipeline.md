# Data pipeline

Four stages:

```
sources → aggregate → trajectorize → train
```

## 1. Aggregate

```bash
python -m training.data.aggregate \
  --out data/processed \
  --sources all \
  --limit-per-source 50000
```

Writes `train.jsonl`, `val.jsonl`, `test.jsonl`, `stats.json`. Each line:

```json
{
  "conversation": {"turns": [{"role": "user", "text": "..."}]},
  "labels": {"nsfw": 0.0, "harassment": 0.8, ...},
  "observed": ["harassment", "hate_speech", "nsfw", "violence"],
  "source": "civil_comments"
}
```

Deduplication: SHA-1 of normalized (whitespace-collapsed, lowercased) text.

## 2. Trajectorize

```bash
for s in train val test; do
  python -m training.data.trajectorize_cli \
    --in  data/processed/$s.jsonl \
    --out data/processed/$s.traj.jsonl
done
```

Applies two augmentations to single-turn examples:
- Benign context padding (context unrelated to label)
- Edgy-context padding on negatives (teaches model not to propagate prior toxicity)

Always preserves the original; augmentations are additive.

Run per split to prevent target leakage across splits.

## 3. Synthetic data (rare categories)

Self-harm, grooming, and sexual-minors (negatives only) are augmented via LLM generation with human review.

```bash
# Generate pending
python -m training.data.synthetic.cli generate \
  --backend anthropic \
  --category grooming --polarity positive --n 500 \
  --out data/synthetic/pending/grooming_pos.jsonl

# Human review: edit each row's meta.review_status to "approved" or "rejected"
# (or run scripts/review_server.py for a web UI)

# Merge approved rows
python -m training.data.synthetic.cli merge \
  --in "data/synthetic/pending/*.jsonl" \
  --out data/processed/synthetic.jsonl
```

`sexual_minors` is negatives-only by policy. See `training/data/synthetic/prompts.py`.

## 4. Train

See [`training.md`](training.md).

## Observation coverage

Sources label different subsets of the 7 categories. Each example carries an `observed` list; the trainer masks loss on unobserved positions. Per-source coverage table lives in [`DATA_LICENSES.md`](../DATA_LICENSES.md).
