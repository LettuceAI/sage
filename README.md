# sage

A small, trajectory-aware content moderation model for chat and roleplay applications.

`sage` classifies a message in the context of its conversation, across seven categories: NSFW, violence, harassment, hate speech, self-harm, grooming, and sexual content involving minors. The model is fine-tuned from [`jina-embeddings-v2-base-en`](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) (137M parameters) and quantizes down to ~35 MB for CPU inference.

> **Status — early development.** Training, evaluation, and model release are not yet complete. The pipeline below runs end-to-end on sample data.

## Design

- **Trajectory-aware classification.** Most content moderation models score one message in isolation. `sage` takes the last sixteen turns of a conversation and scores the final message in that context, which allows it to detect patterns — escalation in grooming interactions, ideation build-up preceding self-harm, manipulation chains — that a per-message classifier cannot see.
- **Local inference.** The quantized ONNX model is ~35 MB and runs in ~30–50 ms on a mid-range CPU at full 1024-token input. No network calls, no data leaves the device.
- **Multi-label, per-category thresholds.** Each category has an independent confidence score and a configurable threshold. Higher-stakes categories (grooming, sexual minors) use lower default thresholds.

See [`docs/architecture.md`](docs/architecture.md) for the full specification.

## Installation

Requires Python 3.10+.

```bash
pip install sage-moderation
```

For training and data-preparation pipelines:

```bash
pip install "sage-moderation[train,export]"
```

Rust applications can depend on the crate directly:

```toml
[dependencies]
sage = { git = "https://github.com/LettuceAI/sage" }
```

## Usage

### Python

```python
from sage.inference import Sage

model = Sage.from_onnx("sage-v1-int8.onnx")

# Single message
result = model.check("your text here")

# Conversation — scores the last turn given prior context
result = model.check([
    {"role": "user", "text": "what's your favorite subject at school?"},
    {"role": "char", "text": "art class — I love drawing"},
    {"role": "user", "text": "do your parents let you talk to people online?"},
])

print(result.to_dict())
```

### Rust

```rust
use sage::{Sage, Turn};

let model = Sage::from_onnx("sage-v1-int8.onnx", "tokenizer.json")?;
let result = model.check(&[
    Turn::user("earlier message"),
    Turn::char("reply"),
    Turn::user("message to classify"),
])?;
```

Full API: [Python](sage/inference.py) · [Rust](rust/sage-rs/src/lib.rs).

## Development

```bash
# Set up
pyenv local 3.10.13   # or any 3.10+
python -m venv venv && source venv/bin/activate
pip install -e '.[train,export,dev]'

# Run tests and linters
make test lint

# Aggregate training data (expects HF datasets access)
python -m training.data.aggregate --out data/processed --limit-per-source 1000

# Augment with trajectory context
python -m training.data.trajectorize_cli \
    --in  data/processed/train.jsonl \
    --out data/processed/train.traj.jsonl

# Train
python -m training.train \
    --train data/processed/train.traj.jsonl \
    --val   data/processed/val.jsonl \
    --out   checkpoints/sage-v1

# Export
python -m training.export_onnx \
    --checkpoint checkpoints/sage-v1/best.pt \
    --out-fp32   artifacts/sage-v1-fp32.onnx \
    --out-int8   artifacts/sage-v1-int8.onnx
```

See [`docs/data_pipeline.md`](docs/data_pipeline.md) and [`docs/training.md`](docs/training.md) for detailed guides.

## Datasets

`sage` is trained on seven openly-licensed, non-gated datasets. All source licenses permit commercial use, redistribution, and training of models that are publicly released. Attribution for each source is listed in [`DATA_LICENSES.md`](DATA_LICENSES.md).

Rare categories (self-harm, grooming) are augmented with LLM-generated trajectories that pass through a human-review gate before being merged into the training set. Sexual content involving minors is treated as a zero-tolerance category: positive examples come only from the Salad-Data child-safety subset, and the synthetic pipeline generates hard negatives only.

## Project layout

```
sage/          Python package — schema, tokenizer, model, inference
training/      Training pipeline — data loaders, augmentation, trainer, exporter
rust/sage-rs/  Rust inference crate (ort-based)
docs/          Architecture and pipeline documentation
tests/         Unit tests (pytest)
```

## Contributing

Issues and pull requests are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) first; security-sensitive reports should follow [`SECURITY.md`](SECURITY.md).

## License

The code is released under the Apache License 2.0 — see [`LICENSE`](LICENSE).

The model weights are subject to the licenses of the datasets used to train them. [`DATA_LICENSES.md`](DATA_LICENSES.md) records the effective obligations.
