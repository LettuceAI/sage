"""Trajectory augmentation for message-level training data.

Two augmentations:
- ``pad_with_benign_context``: adds unrelated prior turns, labels unchanged.
- ``pad_negative_with_edgy_context``: prepends edgy context to clean negatives.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from sage.conversation import Conversation, Role, Turn
from training.data.example import Example

# Fallback chit-chat for when no data-sampled bank is provided.
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
        """Sample benign turns from negative Examples; falls back to defaults."""
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


def _alternating_roles_ending_before(target_role: Role, n: int) -> list[Role]:
    """n roles alternating user/char, ending opposite to target_role."""
    if n <= 0:
        return []
    current = Role.CHAR if target_role is Role.USER else Role.USER
    roles: list[Role] = []
    for _ in range(n):
        roles.append(current)
        current = Role.USER if current is Role.CHAR else Role.CHAR
    roles.reverse()
    return roles


def build_benign_context(
    n_turns: int,
    target_role: Role,
    bank: BenignContextBank,
    rng: random.Random,
    with_system_prompt: bool = False,
) -> list[Turn]:
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


def pad_with_benign_context(
    example: Example,
    bank: BenignContextBank,
    rng: random.Random,
    n_turns: int | None = None,
    with_system_prompt_prob: float = 0.15,
) -> Example:
    """Prepend benign context to a single-turn Example. Labels unchanged."""
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
    """Prepend edgy-but-not-violation context to a pure-negative target."""
    if not example.is_negative():
        raise ValueError(
            "pad_negative_with_edgy_context should only be applied to clean-negative targets"
        )
    if not example.conversation.is_single_message():
        raise ValueError("expected single-turn Example")
    n = n_turns if n_turns is not None else rng.randint(2, 5)
    target = example.target_turn
    roles = _alternating_roles_ending_before(target.role, n)
    context: list[Turn] = []
    for role in roles:
        if edgy_examples and rng.random() < 0.5:
            context.append(Turn(role=role, text=rng.choice(edgy_examples).text))
        else:
            context.append(Turn(role=role, text=bank.sample(role, rng)))
    return Example(
        conversation=Conversation(turns=[*context, target]),
        labels=dict(example.labels),
        observed=example.observed,
        source=example.source + "+edgy_context",
        meta={**example.meta, "augmentation": "edgy_context_negative", "context_turns": n},
    )


@dataclass(slots=True)
class TrajectoryConfig:
    benign_pad_prob: float = 0.6
    edgy_neg_pad_prob: float = 0.25
    preserve_original_prob: float = 0.4
    seed: int = 42


def trajectorize(
    examples: Iterable[Example],
    config: TrajectoryConfig | None = None,
    bank: BenignContextBank | None = None,
) -> Iterator[Example]:
    """Emit a mix of original + augmented Examples; multi-turn inputs pass through."""
    cfg = config or TrajectoryConfig()
    rng = random.Random(cfg.seed)

    pooled: list[Example] = []
    edgy_pool: list[Example] = []
    for ex in examples:
        pooled.append(ex)
        if not ex.is_negative() and len(edgy_pool) < 20_000:
            edgy_pool.append(ex)

    if bank is None:
        bank = BenignContextBank.from_examples(pooled)

    for ex in pooled:
        if not ex.conversation.is_single_message():
            yield ex
            continue

        emit_original = rng.random() < cfg.preserve_original_prob
        if ex.is_negative():
            do_augment = bool(edgy_pool) and rng.random() < cfg.edgy_neg_pad_prob
        else:
            do_augment = rng.random() < cfg.benign_pad_prob

        if emit_original:
            yield ex
        if do_augment:
            try:
                if ex.is_negative():
                    yield pad_negative_with_edgy_context(ex, edgy_pool, bank, rng)
                else:
                    yield pad_with_benign_context(ex, bank, rng)
            except ValueError:
                # Augmentation refused (e.g. multi-turn input slipped through);
                # ensure we still emit the original below.
                do_augment = False

        # Guarantee at least one emission per input so aggregation never
        # silently drops data even when both dice rolls miss.
        if not emit_original and not do_augment:
            yield ex
