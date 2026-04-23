"""Conversation input types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    USER = "user"
    CHAR = "char"
    SYSTEM = "system"


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
    role: Role
    text: str

    @staticmethod
    def from_dict(d: dict) -> Turn:
        return Turn(role=Role(d["role"]), text=d["text"])

    def to_dict(self) -> dict:
        return {"role": self.role.value, "text": self.text}


@dataclass(slots=True)
class Conversation:
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
    def from_text(cls, text: str, role: Role = Role.USER) -> Conversation:
        return cls(turns=[Turn(role=role, text=text)])

    @classmethod
    def from_turns(cls, turns: Iterable[dict | Turn]) -> Conversation:
        return cls(turns=[t if isinstance(t, Turn) else Turn.from_dict(t) for t in turns])

    def to_dict(self) -> dict:
        return {"turns": [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d: dict) -> Conversation:
        return cls.from_turns(d["turns"])


def render_for_debug(conv: Conversation) -> str:
    """Linearize to the token-stream shape for logging."""
    parts: list[str] = ["[CLS]"]
    for turn in conv.context:
        parts.append(f"{ROLE_TOKEN[turn.role]} {turn.text} [SEP]")
    parts.append(f"{CURRENT_TOKEN} {conv.target.text} [SEP]")
    return "\n".join(parts)
