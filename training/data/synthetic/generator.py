"""LLM backends for synthesis. Each exposes a single ``generate(system, user, ...)``."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Generator(ABC):
    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str: ...


class AnthropicGenerator(Generator):
    """Anthropic Messages API backend. Requires ANTHROPIC_API_KEY."""

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "AnthropicGenerator requires the `anthropic` package. "
                "Install with: pip install anthropic"
            ) from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)


class OllamaGenerator(Generator):
    """Ollama HTTP backend (default http://localhost:11434)."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str:
        import requests

        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


@dataclass
class MockGenerator(Generator):
    """Returns queued canned responses, for tests."""

    responses: list[str]

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str:
        if not self.responses:
            raise RuntimeError("MockGenerator response queue is empty")
        return self.responses.pop(0)


def json_loads_lenient(text: str) -> Any:
    """Parse JSON from text that may include code fences or surrounding prose."""
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for open_char, close_char in (("[", "]"), ("{", "}")):
        start = s.find(open_char)
        end = s.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in generator output: {text[:200]!r}")
