# sage

Local multi-label content-moderation model for chat and roleplay.

- 137M parameters (fine-tune of `jinaai/jina-embeddings-v2-base-en`)
- 7 categories: `nsfw`, `violence`, `harassment`, `hate_speech`, `self_harm`, `grooming`, `sexual_minors`
- Accepts a single message or up to 16 turns of conversation; classifies the last turn
- INT8 ONNX artifact ~35 MB, CPU inference ~30–50 ms at 1024 tokens
- Python and Rust inference libraries; no network dependency

**Status — pre-1.0.** Training, evaluation, and first model release are in progress.

## Install

```bash
pip install sage-moderation
# with training/export extras:
pip install "sage-moderation[train,export]"
```

Rust crate:

```toml
[dependencies]
sage = { git = "https://github.com/MegalithOfficial/sage" }
```

## Usage

```python
from sage.inference import Sage

model = Sage.from_onnx("sage-v1-int8.onnx")

model.check("some text")

model.check([
    {"role": "user", "text": "earlier message"},
    {"role": "char", "text": "reply"},
    {"role": "user", "text": "message to classify"},
])
```

Rust:

```rust
use sage::{Sage, Turn};

let model = Sage::from_onnx("sage-v1-int8.onnx", "tokenizer.json")?;
let result = model.check(&[
    Turn::user("earlier message"),
    Turn::char("reply"),
    Turn::user("message to classify"),
])?;
```

Output shape:

```json
{
  "flagged": true,
  "categories": {
    "nsfw":          {"score": 0.03, "flagged": false},
    "violence":      {"score": 0.08, "flagged": false},
    "harassment":    {"score": 0.06, "flagged": false},
    "hate_speech":   {"score": 0.04, "flagged": false},
    "self_harm":     {"score": 0.02, "flagged": false},
    "grooming":      {"score": 0.71, "flagged": true},
    "sexual_minors": {"score": 0.04, "flagged": false}
  }
}
```

## Layout

```
sage/          python package (schema, tokenizer, model, inference, loss)
training/      data pipeline, trainer, onnx exporter
rust/sage-rs/  rust inference crate
benchmarks/    adversarial test suites
notebooks/     colab notebooks for training and synthesis
scripts/       command-line review + eval tools
docs/          architecture and pipeline docs
tests/         pytest
```

## Development

```bash
python -m venv venv
source venv/bin/activate
pip install -e '.[train,export,dev]'
make check
```

## Training pipeline

```bash
python -m training.data.aggregate --out data/processed --sources all --limit-per-source 50000
for s in train val test; do
  python -m training.data.trajectorize_cli --in data/processed/$s.jsonl --out data/processed/$s.traj.jsonl
done
python -m training.train \
  --train data/processed/train.traj.jsonl \
  --val   data/processed/val.traj.jsonl \
  --out   checkpoints/sage-v1
python -m training.export_onnx \
  --checkpoint checkpoints/sage-v1/best.pt \
  --out-fp32 artifacts/sage-v1-fp32.onnx \
  --out-int8 artifacts/sage-v1-int8.onnx
```

See [`docs/training.md`](docs/training.md), [`docs/data_pipeline.md`](docs/data_pipeline.md), [`docs/architecture.md`](docs/architecture.md).

## Training data

Seven openly-licensed, non-gated sources. See [`DATA_LICENSES.md`](DATA_LICENSES.md).

Rare categories are augmented with reviewer-approved LLM-generated conversations. Raw synthetic data is not distributed.

## License

- Code: Apache 2.0 ([`LICENSE`](LICENSE))
- Model weights: subject to constituent dataset licenses
