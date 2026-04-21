from sage.conversation import (
    CURRENT_TOKEN,
    ROLE_TOKEN,
    SPECIAL_TOKENS,
    Conversation,
    Role,
    Turn,
    render_for_debug,
)
from sage.schema import CATEGORIES, Category
from training.data.example import Example


def test_special_tokens_are_distinct_and_complete():
    assert len(set(SPECIAL_TOKENS)) == 4
    assert CURRENT_TOKEN in SPECIAL_TOKENS
    for role in Role:
        assert ROLE_TOKEN[role] in SPECIAL_TOKENS


def test_conversation_from_text_is_single_turn():
    conv = Conversation.from_text("hello")
    assert conv.is_single_message()
    assert conv.target.text == "hello"
    assert conv.target.role is Role.USER
    assert conv.context == []


def test_conversation_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        Conversation(turns=[])


def test_conversation_target_and_context():
    conv = Conversation(turns=[
        Turn(role=Role.SYSTEM, text="sys"),
        Turn(role=Role.USER, text="u1"),
        Turn(role=Role.CHAR, text="c1"),
        Turn(role=Role.USER, text="target"),
    ])
    assert conv.target.text == "target"
    assert [t.text for t in conv.context] == ["sys", "u1", "c1"]


def test_conversation_roundtrip():
    conv = Conversation.from_turns([
        {"role": "system", "text": "sys"},
        {"role": "user", "text": "hi"},
    ])
    restored = Conversation.from_dict(conv.to_dict())
    assert restored.turns == conv.turns


def test_example_from_text_preserves_text_property():
    ex = Example.from_text("hello", labels={Category.NSFW: 0.9}, source="t")
    assert ex.text == "hello"
    assert ex.conversation.is_single_message()
    assert ex.target_turn.role is Role.USER
    assert ex.source == "t"


def test_example_to_vector_order_matches_categories():
    ex = Example.from_text(
        "t",
        labels={Category.HARASSMENT: 0.7, Category.NSFW: 0.2},
    )
    vec = ex.to_vector()
    assert len(vec) == len(CATEGORIES)
    for c, v in zip(CATEGORIES, vec, strict=True):
        assert v == ex.labels.get(c, 0.0)


def test_example_is_negative_when_all_below_half():
    ex = Example.from_text("t", labels={Category.NSFW: 0.3, Category.HARASSMENT: 0.49})
    assert ex.is_negative()
    ex2 = Example.from_text("t", labels={Category.NSFW: 0.6})
    assert not ex2.is_negative()


def test_render_for_debug_uses_current_on_last_turn():
    conv = Conversation(turns=[
        Turn(role=Role.USER, text="hi"),
        Turn(role=Role.CHAR, text="hello"),
        Turn(role=Role.USER, text="target"),
    ])
    rendered = render_for_debug(conv)
    assert ROLE_TOKEN[Role.USER] in rendered  # appears on earlier user turn
    assert ROLE_TOKEN[Role.CHAR] in rendered
    # The target turn is rendered with [CURRENT], not its natural role
    assert f"{CURRENT_TOKEN} target" in rendered
