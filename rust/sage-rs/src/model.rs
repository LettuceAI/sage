//! ONNX session wrapper + public `Sage` API.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ndarray::Array2;
use ort::{inputs, session::Session, value::Value};
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::conversation::{Conversation, Turn};
use crate::error::{Result, SageError};
use crate::schema::{Category, DEFAULT_THRESHOLDS};
use crate::tokenizer::SageTokenizer;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CategoryResult {
    pub score: f32,
    pub flagged: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModerationResult {
    pub flagged: bool,
    pub categories: HashMap<Category, CategoryResult>,
}

pub struct Sage {
    session: Mutex<Session>,
    tokenizer: SageTokenizer,
    thresholds: HashMap<Category, f32>,
}

impl Sage {
    /// Build a `Sage` from an ONNX model file and a HuggingFace `tokenizer.json`.
    ///
    /// The tokenizer must be the same one used at training (typically the
    /// Jina v2 tokenizer, exported via the Python SageTokenizer's base).
    pub fn from_onnx(
        onnx_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::from_onnx_with(onnx_path, tokenizer_path, 1024)
    }

    pub fn from_onnx_with(
        onnx_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        max_length: usize,
    ) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(onnx_path.as_ref())?;
        let inner = Tokenizer::from_file(tokenizer_path.as_ref())?;
        let tokenizer = SageTokenizer::new(inner, max_length)?;
        let thresholds = DEFAULT_THRESHOLDS
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        Ok(Self { session: Mutex::new(session), tokenizer, thresholds })
    }

    pub fn set_threshold(&mut self, category: Category, threshold: f32) {
        self.thresholds.insert(category, threshold);
    }

    /// Classify a single message (1-turn conversation).
    pub fn check_text(&self, text: &str) -> Result<ModerationResult> {
        self.check_conversation(&Conversation::from_text(text))
    }

    /// Classify the last turn of a multi-turn conversation.
    pub fn check(&self, turns: &[Turn]) -> Result<ModerationResult> {
        let conv = Conversation::new(turns.to_vec())?;
        self.check_conversation(&conv)
    }

    pub fn check_conversation(&self, conv: &Conversation) -> Result<ModerationResult> {
        let encoded = self.tokenizer.encode(conv)?;
        let n = self.tokenizer.max_length();

        let input_ids = Array2::from_shape_vec((1, n), encoded.input_ids).expect("shape");
        let attention_mask = Array2::from_shape_vec((1, n), encoded.attention_mask).expect("shape");
        let pooling_mask = Array2::from_shape_vec((1, n), encoded.pooling_mask).expect("shape");

        // Run inference and extract the output tensor into owned data while the
        // session lock is held; then drop the guard before doing the
        // threshold/sigmoid work. `SessionOutputs` borrows from the session, so
        // we can't let it escape the locked scope.
        let (shape_usize, logits_owned): (Vec<usize>, Vec<f32>) = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| SageError::Ort("session mutex poisoned".into()))?;
            let outputs = session.run(inputs![
                "input_ids" => Value::from_array(input_ids)?,
                "attention_mask" => Value::from_array(attention_mask)?,
                "pooling_mask" => Value::from_array(pooling_mask)?,
            ])?;
            let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;
            (
                shape.iter().map(|&d| d as usize).collect(),
                data.to_vec(),
            )
        };

        if shape_usize != [1, 7] {
            return Err(SageError::BadOutputShape {
                expected_batch: 1,
                actual: shape_usize,
            });
        }
        let data = logits_owned.as_slice();

        let mut categories: HashMap<Category, CategoryResult> = HashMap::with_capacity(7);
        let mut any_flagged = false;
        for (i, category) in Category::ALL.iter().enumerate() {
            let score = sigmoid(data[i]);
            let threshold = *self.thresholds.get(category).unwrap_or(&0.5);
            let flagged = score >= threshold;
            any_flagged |= flagged;
            categories.insert(*category, CategoryResult { score, flagged });
        }
        Ok(ModerationResult { flagged: any_flagged, categories })
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
