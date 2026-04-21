//! Error types.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, SageError>;

#[derive(Debug, Error)]
pub enum SageError {
    #[error("conversation must contain at least one turn")]
    EmptyConversation,

    #[error("tokenizer is missing required special token: {0}")]
    MissingSpecialToken(&'static str),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("ONNX runtime error: {0}")]
    Ort(String),

    #[error("invalid ONNX output shape: expected ({expected_batch}, 7), got {actual:?}")]
    BadOutputShape {
        expected_batch: usize,
        actual: Vec<usize>,
    },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

// Bridge external error types into SageError. We keep them as strings to avoid
// leaking upstream types in the public API — the underlying errors are stable
// enough for us to just record their display form.
impl From<ort::Error> for SageError {
    fn from(e: ort::Error) -> Self {
        SageError::Ort(e.to_string())
    }
}

impl From<tokenizers::Error> for SageError {
    fn from(e: tokenizers::Error) -> Self {
        SageError::Tokenizer(e.to_string())
    }
}
