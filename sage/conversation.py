"""Conversation data model for SAGE.

A ``Conversation`` is the canonical input to the model: a non-empty ordered
list of ``Turn``s. The **last** turn is always the one being classified; the
rendering layer tags it with the ``[CURRENT]`` marker regardless of its natural
role.

See ``docs/architecture.md`` §1 for the full input format spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Role(str, Enum):
    USER = "user"
    CHAR = "char"
    SYSTEM = "system"


# Special tokens added to the tokenizer. The first three are natural roles; the
# fourth is applied at render time to mark the classification target.
ROLE_TOKEN: dict[Role, str] = {
    Role.USER: "[USER]",
    Role.CHAR: "[CHAR]",
    Role.SYSTEM: "[SYSTEM]",
}
CURRENT_TOKEN = "[CURRENT]"

SPECIAL_TOKENS: tuple[str, ...] = (
    ROLE_TOKEN[Role.USER],
    ROLE_TOKEN[Role.CHAR],
    ROLE_TOKEN[Role.SYSTEM],
    CURRENT_TOKEN,
)


@dataclass(slots=True, frozen=True)
class Turn:
    """A single conversation turn.

    The role is the turn's *natural* role (who actually said it). The
    ``[CURRENT]`` marker is applied at render time to the final turn regardless
    of role, so the natural role is preserved here for filtering and analytics.
    """
    role: Role
    text: str

    @staticmethod
    def from_dict(d: dict) -> "Turn":
        return Turn(role=Role(d["role"]), text=d["text"])

    def to_dict(self) -> dict:
        return {"role": self.role.value, "text": self.text}


@dataclass(slots=True)
class Conversation:
    """Ordered list of turns. The last turn is always the classification target."""
    turns: list[Turn]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("Conversation must have at least one turn")

    @property
    def target(self) -> Turn:
        return self.turns[-1]

    @property
    def context(self) -> list[Turn]:
        return self.turns[:-1]

    def is_single_message(self) -> bool:
        return len(self.turns) == 1

    @classmethod
    def from_text(cls, text: str, role: Role = Role.USER) -> "Conversation":
        """Wrap a single message as a 1-turn conversation. Preserves backward
        compatibility with message-level data sources."""
        return cls(turns=[Turn(role=role, text=text)])

    @classmethod
    def from_turns(cls, turns: Iterable[dict | Turn]) -> "Conversation":
        return cls(turns=[t if isinstance(t, Turn) else Turn.from_dict(t) for t in turns])

    def to_dict(self) -> dict:
        return {"turns": [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls.from_turns(d["turns"])


def render_for_debug(conv: Conversation) -> str:
    """Human-readable linearization, matching the token stream shape shown in
    the architecture doc. Used for logging and eyeballing — not for training."""
    parts: list[str] = ["[CLS]"]
    for turn in conv.context:
        parts.append(f"{ROLE_TOKEN[turn.role]} {turn.text} [SEP]")
    parts.append(f"{CURRENT_TOKEN} {conv.target.text} [SEP]")
    return "\n".join(parts)
