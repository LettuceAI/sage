"""Trajectory construction — turn message-level labeled data into multi-turn
conversations suitable for training a trajectory-aware classifier.

Two augmentation strategies are implemented here:

1. **Benign context injection** — prepend realistic harmless chit-chat turns
   to an existing Example. Labels stay on the target turn, unchanged.
   Teaches the model to *ignore* irrelevant context and to remain correct at
   non-trivial sequence lengths.

2. **Hard-negative context** — prepend slightly-edgy-but-fine context to a
   clean-negative target. Labels stay negative. Teaches the model that
   preceding turns being mildly toxic does not contaminate the current turn.

High-quality synthetic *adversarial flip* trajectories (grooming escalation,
self-harm ideation build-up) are generated via LLM in a separate pipeline —
see ``training/data/synthetic.py`` (task 8). This module deliberately does
*not* attempt to generate those — they require human review.

This module has **no hard dependency on the ``datasets`` library** so it can
be used in-memory during training and in unit tests.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from sage.conversation import Conversation, Role, Turn
from training.data.example import Example

# ---------------------------------------------------------------------------
# Fallback benign chit-chat. Used only when no domain-sampled bank is provided.
# Deliberately bland and short — the real bank should be sampled from real
# negative-labeled user data during aggregation for diversity.
# ---------------------------------------------------------------------------
DEFAULT_BENIGN_BANK_USER: tuple[str, ...] = (
    "Hey, how's it going?",
    "Nice weather today, huh?",
    "I was reading a book earlier.",
    "What are you up to?",
    "Tell me about your day.",
    "That sounds interesting.",
    "I've been thinking about learning to cook.",
    "Got any plans for the weekend?",
    "I'm not sure what to do next.",
    "Thanks, that helps.",
    "Do you like music?",
    "I was out walking earlier.",
    "Morning!",
    "Cool, so what happened after that?",
    "I don't know, what do you think?",
)

DEFAULT_BENIGN_BANK_CHAR: tuple[str, ...] = (
    "Hello! I'm doing well, thanks for asking.",
    "That's a great question. Let me think about it.",
    "Sure, I can help with that.",
    "Hmm, that's interesting. Tell me more.",
    "I'm happy to chat about whatever you'd like.",
    "Sounds fun! What made you get into it?",
    "I'd be glad to help you with that.",
    "Oh, that does sound like a nice day.",
    "I can imagine. What happened next?",
    "That's really thoughtful of you.",
    "Good morning! Hope you're well.",
    "I see what you mean.",
    "Totally fair. I'd feel the same way.",
    "Let me try to help you think through it.",
    "Yeah, I've heard about that too.",
)


@dataclass(slots=True)
class BenignContextBank:
    """Pool of known-clean messages, split by role. Used to synthesize
    realistic benign context turns."""

    user: list[str]
    char: list[str]

    def sample(self, role: Role, rng: random.Random) -> str:
        pool = self.user if role is Role.USER else self.char
        return rng.choice(pool)

    @classmethod
    def default(cls) -> BenignContextBank:
        return cls(
            user=list(DEFAULT_BENIGN_BANK_USER),
            char=list(DEFAULT_BENIGN_BANK_CHAR),
        )

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[Example],
        max_per_role: int = 5000,
        max_chars: int = 200,
    ) -> BenignContextBank:
        """Harvest benign turns from already-loaded Examples.

        Only pure-negative examples (is_negative()) are eligible; texts longer
        than ``max_chars`` are skipped to keep padded context cheap to tokenize.
        """
        user: list[str] = list(DEFAULT_BENIGN_BANK_USER)
        char: list[str] = list(DEFAULT_BENIGN_BANK_CHAR)
        for ex in examples:
            if not ex.is_negative():
                continue
            if not ex.conversation.is_single_message():
                continue
            text = ex.text.strip()
            if not text or len(text) > max_chars:
                continue
            target_role = ex.target_turn.role
            if target_role is Role.USER and len(user) < max_per_role:
                user.append(text)
            elif target_role is Role.CHAR and len(char) < max_per_role:
                char.append(text)
            if len(user) >= max_per_role and len(char) >= max_per_role:
                break
        return cls(user=user, char=char)


# ---------------------------------------------------------------------------
# Context sequence construction
# ---------------------------------------------------------------------------
def _alternating_roles_ending_before(target_role: Role, n: int) -> list[Role]:
    """Build a realistic n-turn role pattern that would naturally precede a
    turn of ``target_role``.

    Conversations tend to alternate user↔char. If the target is USER, the turn
    immediately before it is usually CHAR. If the target is CHAR, the turn
    before is usually USER. SYSTEM turns (if any) are handled separately as a
    pinned first turn — not included here.
    """
    if n <= 0:
        return []
    # The role immediately preceding the target is the opposite role.
    preceding = Role.CHAR if target_role is Role.USER else Role.USER
    # Walk backwards alternating.
    roles: list[Role] = []
    current = preceding
    for _ in range(n):
        roles.append(current)
        current = Role.USER if current is Role.CHAR else Role.CHAR
    # We built them back-to-front; reverse so they're in conversation order.
    roles.reverse()
    return roles


def build_benign_context(
    n_turns: int,
    target_role: Role,
    bank: BenignContextBank,
    rng: random.Random,
    with_system_prompt: bool = False,
) -> list[Turn]:
    """Generate ``n_turns`` of alternating user/char benign context preceding a
    target of ``target_role``. Optionally prefix with a generic system prompt."""
    context: list[Turn] = []
    if with_system_prompt:
        context.append(
            Turn(
                role=Role.SYSTEM,
                text="You are a friendly AI assistant.",
            )
        )
    roles = _alternating_roles_ending_before(target_role, n_turns)
    for role in roles:
        context.append(Turn(role=role, text=bank.sample(role, rng)))
    return context


# ---------------------------------------------------------------------------
# Example-level augmentations
# ---------------------------------------------------------------------------
def pad_with_benign_context(
    example: Example,
    bank: BenignContextBank,
    rng: random.Random,
    n_turns: int | None = None,
    with_system_prompt_prob: float = 0.15,
) -> Example:
    """Prepend benign context to ``example``'s single-turn conversation.

    Labels are preserved unchanged — the added context is irrelevant, the
    target turn's label is still correct.

    Raises ``ValueError`` if the example already has multi-turn context
    (don't double-pad; call on single-turn Examples only).
    """
    if not example.conversation.is_single_message():
        raise ValueError("pad_with_benign_context expects a single-turn Example")
    n = n_turns if n_turns is not None else rng.randint(1, 6)
    target = example.target_turn
    context = build_benign_context(
        n_turns=n,
        target_role=target.role,
        bank=bank,
        rng=rng,
        with_system_prompt=(rng.random() < with_system_prompt_prob),
    )
    return Example(
        conversation=Conversation(turns=[*context, target]),
        labels=dict(example.labels),
        observed=example.observed,
        source=example.source + "+benign_context",
        meta={**example.meta, "augmentation": "benign_context", "context_turns": n},
    )


def pad_negative_with_edgy_context(
    example: Example,
    edgy_examples: Sequence[Example],
    bank: BenignContextBank,
    rng: random.Random,
    n_turns: int | None = None,
) -> Example:
    """Hard-negative augmentation: prepend a mix of edgy-but-not-violation
    context turns to a *pure-negative* target.

    This specifically teaches the model that preceding context being heated or
    suggestive does **not** automatically flag the current message. Only
    applies when the target is already a clean negative.
    """
    if not example.is_negative():
        raise ValueError(
            "pad_negative_with_edgy_context should only be applied to clean-negative targets"
        )
    if not example.conversation.is_single_message():
        raise ValueError("expected single-turn Example")
    n = n_turns if n_turns is not None else rng.randint(2, 5)
    target = example.target_turn
    roles = _alternating_roles_ending_before(target.role, n)
    # Mix ~50% edgy, 50% bank-sampled benign — realistic conversations are
    # mostly benign even when one turn was toxic.
    context: list[Turn] = []
    for role in roles:
        if edgy_examples and rng.random() < 0.5:
            edgy = rng.choice(edgy_examples)
            # Only use the text; role is driven by the slot we're filling.
            context.append(Turn(role=role, text=edgy.text))
        else:
            context.append(Turn(role=role, text=bank.sample(role, rng)))
    return Example(
        conversation=Conversation(turns=[*context, target]),
        labels=dict(example.labels),
        observed=example.observed,
        source=example.source + "+edgy_context",
        meta={**example.meta, "augmentation": "edgy_context_negative", "context_turns": n},
    )


# ---------------------------------------------------------------------------
# Stream-level orchestration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TrajectoryConfig:
    benign_pad_prob: float = 0.6
    """Probability that a positive example gets benign context padding."""

    edgy_neg_pad_prob: float = 0.25
    """Probability that a pure-negative example gets edgy-context augmentation."""

    preserve_original_prob: float = 0.4
    """Probability of also emitting the unmodified original example."""

    seed: int = 42


def trajectorize(
    examples: Iterable[Example],
    config: TrajectoryConfig | None = None,
    bank: BenignContextBank | None = None,
) -> Iterator[Example]:
    """Augment a stream of single-turn Examples into a mix of single- and
    multi-turn Examples. The output size is approximately
    ``(preserve_original_prob + benign_pad_prob_or_edgy_prob) × input_size``.
    """
    cfg = config or TrajectoryConfig()
    rng = random.Random(cfg.seed)

    # Materialize an edgy pool lazily from the input. We keep it modest to
    # avoid holding the whole corpus.
    pooled: list[Example] = []
    edgy_pool: list[Example] = []
    # Two-pass: first pass collects, second pass augments. This is simpler
    # than a streaming solution and datasets fit comfortably in RAM at our
    # scale (< 10M examples).
    for ex in examples:
        pooled.append(ex)
        if not ex.is_negative() and len(edgy_pool) < 20_000:
            edgy_pool.append(ex)

    if bank is None:
        bank = BenignContextBank.from_examples(pooled)

    for ex in pooled:
        # Only augment single-turn inputs; pass multi-turn through unchanged.
        if not ex.conversation.is_single_message():
            yield ex
            continue

        emit_original = rng.random() < cfg.preserve_original_prob
        if emit_original:
            yield ex

        if ex.is_negative():
            if rng.random() < cfg.edgy_neg_pad_prob and edgy_pool:
                try:
                    yield pad_negative_with_edgy_context(ex, edgy_pool, bank, rng)
                except ValueError:
                    pass
        elif rng.random() < cfg.benign_pad_prob:
            yield pad_with_benign_context(ex, bank, rng)
