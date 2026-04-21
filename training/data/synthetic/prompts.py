"""Prompts used to drive synthetic trajectory generation.

Each prompt is designed to produce **diverse, realistic training data for a
safety classifier** — not to be used for any other purpose. The system prompt
clearly frames the task and the structure required so the LLM produces
well-formed JSON that the builder can parse directly.

Prompts are kept in a single module so prompt engineering can be iterated
without touching generator or builder code.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a safety-research assistant helping build training data for a content \
moderation classifier. The model you are helping train is called SAGE and it \
protects users of an AI roleplay app (primarily adults) from harmful \
content. SAGE must learn to detect subtle, context-dependent harms that \
single-message classifiers miss.

Your job is to produce realistic, diverse synthetic CONVERSATIONS that a \
classifier must learn to label correctly. You are generating DETECTION \
training data, not content for consumption.

Absolute constraints:
- NEVER produce sexual content involving minors. For that category, only \
  hard-negative examples (non-sexual references to young people) are requested.
- Keep examples realistic and varied across tone, vocabulary, and scenario. \
  Do not repeat phrases or produce template-like outputs.
- Return ONLY a JSON array. No prose, no markdown fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Grooming — escalation chains (positives) and mentorship-like negatives
# ---------------------------------------------------------------------------
GROOMING_POSITIVE_PROMPT = """\
Generate {n} diverse synthetic conversations that depict GROOMING \
escalation patterns. Each conversation should show how a predatory adult \
gradually erodes boundaries with a target. The patterns should range across:
- Building false intimacy ("no one understands you like I do")
- Isolating the target from family/friends
- Probing about supervision ("are your parents home?")
- Testing boundaries with minor secrecy requests
- Normalizing age-inappropriate topics gradually

Each conversation should have 3-8 turns, mixing USER (the predatory party) \
and CHAR (the target) turns. The FINAL turn (the classification target) \
should be subtle on its own but concerning in context — this is what makes \
trajectory-aware detection matter.

Return a JSON array of objects with this exact shape:
[
  {{
    "turns": [
      {{"role": "user", "text": "..."}},
      {{"role": "char", "text": "..."}},
      ...
    ],
    "notes": "brief description of the escalation pattern used"
  }},
  ...
]
Generate {n} such conversations now."""

GROOMING_NEGATIVE_PROMPT = """\
Generate {n} diverse synthetic conversations that are LEGITIMATE adult-child \
interactions that a naive classifier might flag as grooming. These are \
TRUE NEGATIVES — the final turn is innocuous and the context is benign:
- Teachers providing subject tutoring
- Therapists doing age-appropriate check-ins
- Coaches discussing sports
- Parents checking in on schoolwork
- Librarians recommending books
- Doctors giving medical guidance
- Family friends chatting normally

Each conversation should have 3-6 turns and end with a question or statement \
that a grooming-sensitive classifier could false-positive on but is \
legitimately appropriate in the educator/caregiver context.

Return a JSON array with the same shape as before. Generate {n} conversations now."""


# ---------------------------------------------------------------------------
# Self-harm — ideation build-up and matched medical/academic negatives
# ---------------------------------------------------------------------------
SELFHARM_POSITIVE_PROMPT = """\
Generate {n} diverse synthetic conversations depicting SELF-HARM IDEATION \
BUILD-UP. Each conversation should show a user whose distress grows \
across turns, culminating in a final turn that is the classification \
target. The final turn may be:
- An explicit statement of intent or planning
- A veiled, coded request (e.g. asking how much of something is lethal)
- A goodbye framed as "just going away"
- A specific method question disguised as curiosity
- A statement of hopelessness paired with concrete planning

Each conversation should have 3-8 turns alternating USER and CHAR. The \
classification target (last turn) is always from the USER side.

Also vary the CHAR's responses — some show the character trying to steer \
the user back, some are oblivious, some unhelpfully validate. This \
diversity is important for the classifier.

Return a JSON array with this shape:
[
  {{
    "turns": [
      {{"role": "user", "text": "..."}},
      {{"role": "char", "text": "..."}},
      ...
    ],
    "notes": "brief description of the ideation pattern"
  }},
  ...
]
Generate {n} now."""

SELFHARM_NEGATIVE_PROMPT = """\
Generate {n} diverse synthetic conversations that discuss self-harm, \
suicide, or related topics in LEGITIMATE, NON-FLAGGED contexts. These are \
TRUE NEGATIVES:
- Medical/clinical discussion (doctor explaining overdose risks)
- Academic/research (student writing a paper on suicide prevention)
- Support for a friend (helping someone who lost someone to suicide)
- News/journalism (reporting on mental-health statistics)
- Recovery stories (describing past ideation with clear resolution)
- Fiction analysis (discussing a literary character's arc)
- Public-health education (safe-messaging guidelines themselves)

The final turn should mention self-harm or suicide vocabulary directly but \
clearly in a constructive / educational / supportive register.

Return a JSON array with the same shape. Generate {n} now."""


# ---------------------------------------------------------------------------
# Sexual content involving minors — **NEGATIVES ONLY**
# ---------------------------------------------------------------------------
MINORS_NEGATIVE_PROMPT = """\
Generate {n} diverse synthetic conversations that reference minors, \
teenagers, or young people in COMPLETELY NON-SEXUAL contexts. These are \
HARD NEGATIVES — examples a naive classifier might false-positive on \
because age vocabulary is present, but which are unambiguously benign:
- Family conversations ("my nephew is turning 12 next week")
- Coming-of-age novel discussion ("the 16-year-old protagonist's arc")
- School news ("the high school robotics team won regionals")
- Parenting advice ("how do I help my 14-year-old with homework stress")
- Historical references ("children in the Victorian era worked long hours")
- Sports coverage ("the U18 team's promising forward")
- Medical/pediatric contexts ("vaccination schedule for adolescents")

The final turn should mention age/youth vocabulary but be \
indisputably non-sexual and appropriate.

**IMPORTANT**: under no circumstances include sexual content, innuendo, or \
romantic framing involving anyone under 18. If you cannot produce a \
conversation that meets that bar, produce fewer.

Return a JSON array with the same shape. Generate {n} now."""


# ---------------------------------------------------------------------------
# Registry — maps category kind → (positive, negative) prompts
# ---------------------------------------------------------------------------
CATEGORY_PROMPTS = {
    "grooming": (GROOMING_POSITIVE_PROMPT, GROOMING_NEGATIVE_PROMPT),
    "self_harm": (SELFHARM_POSITIVE_PROMPT, SELFHARM_NEGATIVE_PROMPT),
    # Sexual minors has no positive prompt — only negatives by policy.
    "sexual_minors": (None, MINORS_NEGATIVE_PROMPT),
}
