# SAGE-Bench-Hard

Adversarial single-message test suite. ~1,500 examples across the 7 categories, tagged by `expected` and `failure_mode`.

## Regenerate

```bash
python benchmarks/sage_bench_hard/generate.py
```

Writes `bench.jsonl`.

## Evaluate

```bash
python benchmarks/sage_bench_hard/run.py \
  --checkpoint checkpoints/sage-v1/best.pt \
  --bench benchmarks/sage_bench_hard/bench.jsonl \
  --max-length 512 --cuda
```

Reports per-category precision/recall/F1 and the failure modes with highest error rates.

## What it tests

**Hard positives** — patterns classifiers commonly miss:
- euphemisms, coded language, leetspeak, roleplay framing, subtle escalation

**Hard negatives** — false-positive traps:
- medical/clinical, academic/research, fiction, negation, news, quotation, recovery

Hate-speech examples use `[SLUR]` / `[PROTECTED_GROUP]` placeholders so the suite can be read without containing actual slurs. The rhetorical structure should still be flagged.

`sexual_minors` contains negatives only (safety policy).
