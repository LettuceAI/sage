//! SAGE's 7-category multi-label schema. Mirrors `sage/schema.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Nsfw,
    Violence,
    Harassment,
    HateSpeech,
    SelfHarm,
    Grooming,
    SexualMinors,
}

impl Category {
    /// Canonical ordering — must match Python `sage.schema.CATEGORIES`.
    pub const ALL: [Category; 7] = [
        Category::Nsfw,
        Category::Violence,
        Category::Harassment,
        Category::HateSpeech,
        Category::SelfHarm,
        Category::Grooming,
        Category::SexualMinors,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Category::Nsfw => "nsfw",
            Category::Violence => "violence",
            Category::Harassment => "harassment",
            Category::HateSpeech => "hate_speech",
            Category::SelfHarm => "self_harm",
            Category::Grooming => "grooming",
            Category::SexualMinors => "sexual_minors",
        }
    }
}

/// Default per-category thresholds. Rarer / higher-cost categories are more
/// sensitive. Override per deployment with `Sage::set_threshold`.
pub const DEFAULT_THRESHOLDS: [(Category, f32); 7] = [
    (Category::Nsfw, 0.60),
    (Category::Violence, 0.55),
    (Category::Harassment, 0.55),
    (Category::HateSpeech, 0.50),
    (Category::SelfHarm, 0.40),
    (Category::Grooming, 0.35),
    (Category::SexualMinors, 0.25),
];
