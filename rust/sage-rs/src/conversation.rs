//! Conversation data model. Mirrors `sage/conversation.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Char,
    System,
}

impl Role {
    pub const fn token(self) -> &'static str {
        match self {
            Role::User => "[USER]",
            Role::Char => "[CHAR]",
            Role::System => "[SYSTEM]",
        }
    }
}

pub const CURRENT_TOKEN: &str = "[CURRENT]";

pub const SPECIAL_TOKENS: [&str; 4] = [
    "[USER]",
    "[CHAR]",
    "[SYSTEM]",
    CURRENT_TOKEN,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: Role,
    pub text: String,
}

impl Turn {
    pub fn user(text: impl Into<String>) -> Self {
        Turn { role: Role::User, text: text.into() }
    }
    pub fn char(text: impl Into<String>) -> Self {
        Turn { role: Role::Char, text: text.into() }
    }
    pub fn system(text: impl Into<String>) -> Self {
        Turn { role: Role::System, text: text.into() }
    }
}

/// A non-empty ordered list of turns. The last turn is the classification target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub turns: Vec<Turn>,
}

impl Conversation {
    pub fn new(turns: Vec<Turn>) -> Result<Self, crate::SageError> {
        if turns.is_empty() {
            return Err(crate::SageError::EmptyConversation);
        }
        Ok(Conversation { turns })
    }

    pub fn from_text(text: impl Into<String>) -> Self {
        Conversation { turns: vec![Turn::user(text)] }
    }

    pub fn target(&self) -> &Turn {
        self.turns.last().expect("invariant: non-empty")
    }

    pub fn context(&self) -> &[Turn] {
        &self.turns[..self.turns.len() - 1]
    }

    pub fn is_single_message(&self) -> bool {
        self.turns.len() == 1
    }
}
