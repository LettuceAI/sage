# sage (Rust)

Rust inference crate for the [SAGE](https://github.com/LettuceAI/sage) content moderation model.

- Loads an ONNX SAGE model + HuggingFace tokenizer
- Encodes trajectory-aware conversation inputs
- Returns 7-category scores + flagged booleans
- Zero runtime allocations for common paths

## Usage

```rust
use sage::{Sage, Turn, Role};

let sage = Sage::from_onnx("sage-v1-int8.onnx", "tokenizer.json")?;

// Single message
let result = sage.check_text("your text here")?;

// Conversation-aware
let result = sage.check(&[
    Turn::user("earlier message"),
    Turn::char("reply"),
    Turn::user("message to classify"),  // last turn is scored
])?;

for (category, r) in &result.categories {
    println!("{:?}: score={:.3} flagged={}", category, r.score, r.flagged);
}
```

## Artifacts

Export from Python with `python -m training.export_onnx` then ship the `.onnx` and
`tokenizer.json` files alongside your app. See the repo root README for the full pipeline.
