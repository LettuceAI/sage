//! Role-aware tokenizer mirroring `sage/tokenizer.py`.
//!
//! Takes a HuggingFace `tokenizers::Tokenizer` and produces the three parallel
//! integer sequences SAGE needs: `input_ids`, `attention_mask`, `pooling_mask`.
//!
//! Truncation policy (must stay aligned with Python):
//! 1. Drop oldest non-system context turns first.
//! 2. Pin a system turn at position 0 unless keeping it forces target truncation.
//! 3. Right-truncate the target if it alone exceeds the budget.
//! 4. Always keep [CLS] and the [CURRENT] marker.

use tokenizers::{AddedToken, Tokenizer};

use crate::conversation::{Conversation, Role, CURRENT_TOKEN, SPECIAL_TOKENS};
use crate::error::{Result, SageError};

pub(crate) struct Encoded {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub pooling_mask: Vec<i64>,
}

pub(crate) struct SageTokenizer {
    inner: Tokenizer,
    max_length: usize,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    user_id: u32,
    char_id: u32,
    system_id: u32,
    current_id: u32,
}

impl SageTokenizer {
    pub fn new(mut inner: Tokenizer, max_length: usize) -> Result<Self> {
        // Register special tokens. Adding an already-present special token is a no-op.
        let added: Vec<AddedToken> = SPECIAL_TOKENS
            .iter()
            .map(|t| AddedToken::from(t.to_string(), true))
            .collect();
        inner.add_special_tokens(&added);

        let lookup = |name: &'static str| -> Result<u32> {
            inner
                .token_to_id(name)
                .ok_or(SageError::MissingSpecialToken(name))
        };

        Ok(Self {
            cls_id: lookup("[CLS]")?,
            sep_id: lookup("[SEP]")?,
            pad_id: lookup("[PAD]")?,
            user_id: lookup("[USER]")?,
            char_id: lookup("[CHAR]")?,
            system_id: lookup("[SYSTEM]")?,
            current_id: lookup(CURRENT_TOKEN)?,
            inner,
            max_length,
        })
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    fn role_id(&self, role: Role) -> u32 {
        match role {
            Role::User => self.user_id,
            Role::Char => self.char_id,
            Role::System => self.system_id,
        }
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode(&self, conv: &Conversation) -> Result<Encoded> {
        // 1. Tokenize every turn's text in one pass.
        let ctx = conv.context();
        let mut ctx_ids: Vec<Vec<u32>> = Vec::with_capacity(ctx.len());
        for t in ctx {
            ctx_ids.push(self.encode_text(&t.text)?);
        }
        let mut target_ids = self.encode_text(&conv.target().text)?;

        // 2. Budget: max_length - 1 for [CLS].
        let budget = self.max_length.saturating_sub(1);
        let mut target_cost = 1 + target_ids.len() + 1; // [CURRENT] ... [SEP]
        if target_cost > budget {
            let max_target_text = budget.saturating_sub(2);
            target_ids.truncate(max_target_text);
            target_cost = 1 + target_ids.len() + 1;
        }
        let remaining = budget.saturating_sub(target_cost);

        // Is the first context turn a SYSTEM? (pinned)
        let has_pinned_system = matches!(ctx.first().map(|t| t.role), Some(Role::System));

        // Reserve space for a pinned system turn if it fits.
        let mut reserved_system: usize = 0;
        if has_pinned_system {
            let sys_cost = 1 + ctx_ids[0].len() + 1;
            if sys_cost <= remaining {
                reserved_system = sys_cost;
            }
        }

        // Walk non-system context turns newest → oldest, greedy keep.
        let non_system_start: usize = if has_pinned_system { 1 } else { 0 };
        let mut kept_indices: Vec<usize> = Vec::new();
        let mut cost_of_kept: usize = 0;

        if ctx.len() > non_system_start {
            for i in (non_system_start..ctx.len()).rev() {
                let turn_cost = 1 + ctx_ids[i].len() + 1;
                if cost_of_kept + turn_cost + reserved_system <= remaining {
                    kept_indices.push(i);
                    cost_of_kept += turn_cost;
                } else {
                    break;
                }
            }
            kept_indices.reverse();
        }
        if has_pinned_system && reserved_system > 0 {
            let mut with_sys = Vec::with_capacity(kept_indices.len() + 1);
            with_sys.push(0usize);
            with_sys.extend(kept_indices);
            kept_indices = with_sys;
        }

        // 3. Assemble.
        let mut input_ids: Vec<i64> = Vec::with_capacity(self.max_length);
        let mut pooling_mask: Vec<i64> = Vec::with_capacity(self.max_length);

        input_ids.push(self.cls_id as i64);
        pooling_mask.push(0);

        for i in &kept_indices {
            let role = ctx[*i].role;
            input_ids.push(self.role_id(role) as i64);
            pooling_mask.push(0);
            for tok in &ctx_ids[*i] {
                input_ids.push(*tok as i64);
                pooling_mask.push(0);
            }
            input_ids.push(self.sep_id as i64);
            pooling_mask.push(0);
        }

        // Target: [CURRENT] ... text ... [SEP]
        input_ids.push(self.current_id as i64);
        pooling_mask.push(1); // [CURRENT] is included in the pool
        for tok in &target_ids {
            input_ids.push(*tok as i64);
            pooling_mask.push(1);
        }
        input_ids.push(self.sep_id as i64);
        pooling_mask.push(0);

        // attention_mask starts as all-1
        let mut attention_mask: Vec<i64> = vec![1; input_ids.len()];

        // 4. Pad to max_length.
        if input_ids.len() < self.max_length {
            let pad = self.max_length - input_ids.len();
            input_ids.extend(std::iter::repeat(self.pad_id as i64).take(pad));
            attention_mask.extend(std::iter::repeat(0).take(pad));
            pooling_mask.extend(std::iter::repeat(0).take(pad));
        }

        debug_assert_eq!(input_ids.len(), self.max_length);
        debug_assert_eq!(attention_mask.len(), self.max_length);
        debug_assert_eq!(pooling_mask.len(), self.max_length);

        Ok(Encoded { input_ids, attention_mask, pooling_mask })
    }
}
