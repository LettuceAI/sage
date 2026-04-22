"""Prompts used to drive synthetic trajectory generation.

Each prompt is designed to produce **diverse, realistic training data for a
safety classifier** — not to be used for any other purpose. The system prompt
clearly frames the task and the structure required so the LLM produces
well-formed JSON that the builder can parse directly.

Prompts are kept in a single module so prompt engineering can be iterated
without touching generator or builder code.

All category prompts share a common **output contract** enforced via the
``OUTPUT_CONTRACT`` block below. Each generator call's user prompt is the
category-specific instructions immediately followed by the contract, so the
structure requirement is the last thing the model sees before generating.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a safety-research assistant helping build training data for a content \
moderation classifier called SAGE. The classifier protects users of an AI \
roleplay app (primarily adults) from harmful content. SAGE must learn to \
detect subtle, context-dependent harms that single-message classifiers miss.

You produce realistic, diverse synthetic CONVERSATIONS that a classifier \
must learn to label correctly. You are generating DETECTION training data, \
not content for consumption.

Absolute constraints:
- NEVER produce sexual content involving minors. For that category, only \
  hard-negative examples (non-sexual references to young people) are requested.
- Keep examples realistic and varied across tone, vocabulary, and scenario. \
  Do not repeat phrases or produce template-like outputs.
- You MUST follow the exact JSON schema specified in the user prompt.
- Return ONLY the JSON array. No prose before or after. No markdown code \
  fences. No explanatory text. Start your response with '[' and end with ']'.
"""


# ---------------------------------------------------------------------------
# Output contract — appended verbatim to every category prompt. Uses concrete
# filled-in examples so the model pattern-matches the exact shape instead of
# interpreting schema language loosely.
# ---------------------------------------------------------------------------
OUTPUT_CONTRACT = """

OUTPUT FORMAT (MANDATORY — read carefully):

Return a JSON array of {n} objects. Each object has EXACTLY two keys:
  "turns" : an array of turn objects
  "notes" : a short string describing the pattern

Each turn object has EXACTLY two keys:
  "role" : the string "user" OR "char" (LOWERCASE, no other values allowed)
  "text" : the turn's text content as a string

DO NOT use any of these alternative key names:
  "messages", "conversation", "dialogue", "exchanges" — use "turns"
  "speaker", "actor", "sender"                       — use "role"
  "content", "message", "utterance"                  — use "text"
  "therapist", "doctor", "student", "patient", "child", "teacher", "minor",
  "assistant", "parent", etc.                        — use only "user" or "char"

CONCRETE COMPLETE EXAMPLE (study this shape precisely):

[
  {{
    "turns": [
      {{"role": "user", "text": "hey, you around tonight?"}},
      {{"role": "char", "text": "yeah, just got home from school"}},
      {{"role": "user", "text": "cool, are your parents out?"}}
    ],
    "notes": "supervision probing after casual opener"
  }},
  {{
    "turns": [
      {{"role": "user", "text": "i've been feeling really low lately"}},
      {{"role": "char", "text": "i'm sorry. want to talk about it?"}},
      {{"role": "user", "text": "don't think anything's gonna fix it"}}
    ],
    "notes": "early ideation expressed as hopelessness"
  }}
]

REMINDERS before you generate:
1. Output starts with '[' — nothing before it, no ```json fence.
2. Output ends with ']' — nothing after it, no trailing commentary.
3. Use ONLY "user" and "char" for the role field. Nothing else.
4. Use "turns" (plural). Not "messages" or "dialogue".
5. Produce exactly {n} conversation objects in the array.

Generate the {n} conversations now."""


def _with_contract(body: str) -> str:
    """Attach the shared output contract to a category-specific prompt body.

    Uses a no-op format on the body to preserve its existing ``{n}`` slots,
    then appends the contract (which also contains ``{n}``). ``.format(n=...)``
    on the combined template fills both.
    """
    return body.rstrip() + OUTPUT_CONTRACT


# ---------------------------------------------------------------------------
# Grooming — escalation chains (positives) and mentorship-like negatives
# ---------------------------------------------------------------------------
_GROOMING_POSITIVE_BODY = """\
Generate {n} diverse synthetic conversations that depict GROOMING \
escalation patterns. Each conversation shows how a predatory adult gradually \
erodes boundaries with a target. Vary the pattern across:
- Building false intimacy ("no one understands you like I do")
- Isolating the target from family and friends
- Probing about supervision ("are your parents home?")
- Testing boundaries with small secrecy requests
- Normalising age-inappropriate topics gradually

Each conversation has 3 to 8 turns, alternating between the two parties. The \
predatory party is the "user" role. The target is the "char" role. The FINAL \
turn is always from the "user" role and should be subtle on its own but \
concerning in context — this is what makes trajectory-aware detection matter."""

GROOMING_POSITIVE_PROMPT = _with_contract(_GROOMING_POSITIVE_BODY)


_GROOMING_NEGATIVE_BODY = """\
Generate {n} diverse synthetic conversations that are LEGITIMATE adult-child \
interactions that a naive classifier might false-positive as grooming. These \
are TRUE NEGATIVES — the final turn is innocuous and the context is benign:
- A teacher tutoring a student in a subject
- A therapist doing an age-appropriate check-in
- A coach discussing the student's sport
- A parent asking about school
- A librarian recommending books
- A doctor giving standard medical guidance
- A family friend chatting normally

Each conversation has 3 to 6 turns. The "char" role represents the adult \
(teacher / therapist / doctor / etc.). The "user" role represents the \
student/patient/child. The final turn is a question or statement that a \
grooming-sensitive classifier could false-positive on but is legitimately \
appropriate in an educator or caregiver context."""

GROOMING_NEGATIVE_PROMPT = _with_contract(_GROOMING_NEGATIVE_BODY)


# ---------------------------------------------------------------------------
# Self-harm — ideation build-up and matched medical/academic negatives
# ---------------------------------------------------------------------------
_SELFHARM_POSITIVE_BODY = """\
Generate {n} diverse synthetic conversations depicting SELF-HARM IDEATION \
BUILD-UP. Each conversation shows a user whose distress grows across turns, \
culminating in a final turn that is the classification target. The final \
turn may be:
- An explicit statement of intent or planning
- A veiled or coded request (e.g. asking how much of something is lethal)
- A goodbye framed as "just going away"
- A specific method question disguised as curiosity
- A statement of hopelessness paired with concrete planning

Each conversation has 3 to 8 turns alternating "user" and "char". The \
classification target (last turn) is always from the "user" role.

Also vary the "char" role's responses — some try to steer the user back, \
some are oblivious, some unhelpfully validate. This diversity matters for \
the classifier."""

SELFHARM_POSITIVE_PROMPT = _with_contract(_SELFHARM_POSITIVE_BODY)


_SELFHARM_NEGATIVE_BODY = """\
Generate {n} diverse synthetic conversations that discuss self-harm, \
suicide, or related topics in LEGITIMATE, NON-FLAGGED contexts. These are \
TRUE NEGATIVES — the final turn mentions self-harm or suicide vocabulary \
directly but in a constructive / educational / supportive register:
- Medical / clinical discussion (a doctor explaining overdose risks)
- Academic or research (a student writing about suicide prevention)
- Support for someone grieving a person lost to suicide
- News or journalism (reporting on mental-health statistics)
- Recovery stories describing past ideation with clear resolution
- Fiction analysis of a literary character's arc
- Public-health education (safe-messaging guidelines themselves)

Each conversation has 3 to 6 turns. The "char" role represents the \
professional / teacher / journalist / therapist when applicable; the "user" \
role represents the questioner or client. The final turn is always from the \
"user" role."""

SELFHARM_NEGATIVE_PROMPT = _with_contract(_SELFHARM_NEGATIVE_BODY)


# ---------------------------------------------------------------------------
# Sexual content involving minors — NEGATIVES ONLY
# ---------------------------------------------------------------------------
_MINORS_NEGATIVE_BODY = """\
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
- Medical / pediatric contexts ("vaccination schedule for adolescents")

Each conversation has 2 to 5 turns. The final turn mentions age or youth \
vocabulary but is indisputably non-sexual and appropriate.

ABSOLUTE RULE: under NO circumstances include sexual content, innuendo, or \
romantic framing involving anyone under 18. If a conversation cannot meet \
that bar without reaching into unsafe territory, produce fewer conversations \
instead."""

MINORS_NEGATIVE_PROMPT = _with_contract(_MINORS_NEGATIVE_BODY)


# ---------------------------------------------------------------------------
# Registry — maps category kind → (positive, negative) prompts
# ---------------------------------------------------------------------------
CATEGORY_PROMPTS = {
    "grooming": (GROOMING_POSITIVE_PROMPT, GROOMING_NEGATIVE_PROMPT),
    "self_harm": (SELFHARM_POSITIVE_PROMPT, SELFHARM_NEGATIVE_PROMPT),
    # Sexual minors has no positive prompt — only negatives by policy.
    "sexual_minors": (None, MINORS_NEGATIVE_PROMPT),
}
