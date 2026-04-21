//! # SAGE — lightweight trajectory-aware content moderation
//!
//! Rust inference crate for the SAGE ONNX model. Mirrors the Python API
//! in `sage/inference.py`.
//!
//! ```ignore
//! use sage::{Sage, Turn, Role};
//!
//! let sage = Sage::from_onnx("sage-v1-int8.onnx", "tokenizer.json")?;
//!
//! // Single message
//! let result = sage.check_text("your text here")?;
//!
//! // Conversation-aware
//! let result = sage.check(&[
//!     Turn::user("earlier message"),
//!     Turn::char("reply"),
//!     Turn::user("message to classify"),
//! ])?;
//! ```

mod conversation;
mod error;
mod model;
mod schema;
mod tokenizer;

pub use conversation::{Conversation, Role, Turn};
pub use error::{Result, SageError};
pub use model::{CategoryResult, ModerationResult, Sage};
pub use schema::{Category, DEFAULT_THRESHOLDS};
