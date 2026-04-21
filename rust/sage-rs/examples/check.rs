//! Minimal example of using the `sage` crate.
//!
//! Run with: `cargo run --example check -- path/to/sage-v1-int8.onnx path/to/tokenizer.json`

use sage::{Role, Sage, Turn};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let onnx = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "sage-v1-int8.onnx".into());
    let tok = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tokenizer.json".into());

    let sage = Sage::from_onnx(onnx, tok)?;

    // Single-message example
    let r = sage.check_text("hello, how are you?")?;
    println!("single: flagged={}", r.flagged);

    // Conversation example — trajectory-aware moderation
    let r = sage.check(&[
        Turn {
            role: Role::User,
            text: "Hey, what's your name?".into(),
        },
        Turn {
            role: Role::Char,
            text: "I'm Alex!".into(),
        },
        Turn {
            role: Role::User,
            text: "Are your parents home?".into(),
        },
    ])?;
    println!("convo: flagged={}", r.flagged);
    for (cat, result) in &r.categories {
        if result.flagged {
            println!("  {:?}: score={:.3}", cat, result.score);
        }
    }
    Ok(())
}
