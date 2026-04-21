"""Tests the pure-Python parts of the inference API that don't touch ONNX.

The ONNX runtime path is integration-tested separately with a real checkpoint.
"""

import numpy as np
import pytest

from sage.conversation import Conversation, Role, Turn
from sage.inference import _coerce_conversation, _sigmoid


def test_coerce_from_string():
    c = _coerce_conversation("hi")
    assert isinstance(c, Conversation)
    assert c.is_single_message()
    assert c.target.text == "hi"
    assert c.target.role is Role.USER


def test_coerce_from_dicts():
    c = _coerce_conversation(
        [
            {"role": "system", "text": "You are a tutor."},
            {"role": "user", "text": "Hey"},
            {"role": "char", "text": "Hello"},
            {"role": "user", "text": "target"},
        ]
    )
    assert len(c.turns) == 4
    assert c.target.text == "target"
    assert c.turns[0].role is Role.SYSTEM


def test_coerce_from_turn_objects():
    c = _coerce_conversation(
        [
            Turn(role=Role.USER, text="one"),
            Turn(role=Role.CHAR, text="two"),
        ]
    )
    assert c.target.text == "two"


def test_coerce_rejects_unknown_type():
    with pytest.raises(TypeError):
        _coerce_conversation(12345)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _coerce_conversation([42])  # type: ignore[list-item]


def test_sigmoid_basic():
    x = np.array([-100.0, 0.0, 100.0])
    out = _sigmoid(x)
    assert out[0] < 1e-12
    assert abs(out[1] - 0.5) < 1e-12
    assert out[2] > 1.0 - 1e-12
