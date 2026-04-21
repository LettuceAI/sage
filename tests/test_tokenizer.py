"""Offline tokenizer tests using a hand-built WordPiece stub.

Avoids downloading the Jina v2 tokenizer during tests — we construct a tiny
tokenizer with just the behaviors SageTokenizer actually depends on.
"""
from dataclasses import dataclass, field

import pytest

from sage.conversation import (
    CURRENT_TOKEN,
    ROLE_TOKEN,
    SPECIAL_TOKENS,
    Conversation,
    Role,
    Turn,
)
from sage.tokenizer import SageTokenizer


@dataclass
class StubTokenizer:
    """Tiny deterministic tokenizer that behaves like a HuggingFace tokenizer
    for the methods SageTokenizer uses: encode, convert_tokens_to_ids,
    add_special_tokens, __len__, cls_token_id, sep_token_id, pad_token_id,
    unk_token_id."""
    _vocab: dict[str, int] = field(default_factory=lambda: {
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
    })
    cls_token_id: int = 2
    sep_token_id: int = 3
    pad_token_id: int = 0
    unk_token_id: int = 1

    def __len__(self) -> int:
        return len(self._vocab)

    def _intern(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def add_special_tokens(self, extras: dict) -> int:
        added = 0
        for tok in extras.get("additional_special_tokens", []):
            if tok not in self._vocab:
                self._intern(tok)
                added += 1
        return added

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._vocab.get(tokens, self.unk_token_id)
        return [self._vocab.get(t, self.unk_token_id) for t in tokens]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Whitespace word-level tokenization, interning new tokens.
        ids = [self._intern(w) for w in text.split()]
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
        return ids


def _make_tokenizer(max_length: int = 32) -> SageTokenizer:
    return SageTokenizer(tokenizer=StubTokenizer(), max_length=max_length)


def test_special_tokens_are_registered():
    tok = _make_tokenizer()
    assert tok.added_token_count == 4
    for s in SPECIAL_TOKENS:
        assert tok.tokenizer.convert_tokens_to_ids(s) != tok.tokenizer.unk_token_id


def test_encode_single_message_has_correct_structure():
    tok = _make_tokenizer(max_length=16)
    conv = Conversation.from_text("hello world")
    enc = tok.encode(conv)

    # Expected layout (pre-padding):
    # [CLS] [CURRENT] hello world [SEP]
    assert enc.input_ids[0] == tok.cls_id
    assert enc.input_ids[1] == tok.current_id
    # The terminating [SEP] should be at length-1 before padding
    length = enc.length
    assert enc.input_ids[length - 1] == tok.sep_id

    # Pooling mask should cover [CURRENT] + "hello" + "world" = 3 positions
    pooled_positions = [i for i, m in enumerate(enc.pooling_mask) if m]
    assert pooled_positions == [1, 2, 3]


def test_encode_multi_turn_pools_only_current():
    tok = _make_tokenizer(max_length=64)
    conv = Conversation(turns=[
        Turn(role=Role.USER, text="hey"),
        Turn(role=Role.CHAR, text="hi there"),
        Turn(role=Role.USER, text="target message"),
    ])
    enc = tok.encode(conv)

    # At least one pooled position
    assert any(enc.pooling_mask)

    # All pooled positions must fall in the tail of the sequence, and the
    # position immediately before the first pooled token must be the [CURRENT]
    # token — the [CURRENT] marker is itself inside the pooling mask.
    pooled = [i for i, m in enumerate(enc.pooling_mask) if m]
    # The first pooled index should be a [CURRENT] token
    assert enc.input_ids[pooled[0]] == tok.current_id
    # All pooled tokens are contiguous
    assert pooled == list(range(pooled[0], pooled[-1] + 1))


def test_truncation_drops_oldest_context_first():
    # Set max_length so only 1-2 context turns fit.
    tok = _make_tokenizer(max_length=12)
    # Four context turns + target = 5 turns total. The oldest context
    # (non-system) should be dropped first.
    conv = Conversation(turns=[
        Turn(role=Role.USER, text="one two three"),       # should be dropped
        Turn(role=Role.CHAR, text="four"),
        Turn(role=Role.USER, text="five"),
        Turn(role=Role.CHAR, text="six"),
        Turn(role=Role.USER, text="target"),
    ])
    enc = tok.encode(conv)
    # Target must always be present
    current_positions = [i for i, tid in enumerate(enc.input_ids) if tid == tok.current_id]
    assert len(current_positions) == 1
    # Oldest turn's words should not appear
    assert tok.tokenizer._vocab.get("one") not in set(enc.input_ids)


def test_pinned_system_turn_is_preserved():
    tok = _make_tokenizer(max_length=20)
    conv = Conversation(turns=[
        Turn(role=Role.SYSTEM, text="system prompt"),
        Turn(role=Role.USER, text="a"),
        Turn(role=Role.CHAR, text="b"),
        Turn(role=Role.USER, text="target"),
    ])
    enc = tok.encode(conv)
    system_id = tok.role_id[Role.SYSTEM]
    assert system_id in enc.input_ids


def test_oversize_target_is_right_truncated():
    tok = _make_tokenizer(max_length=8)
    conv = Conversation.from_text("a b c d e f g h i j k l m n o p")
    enc = tok.encode(conv)
    # The first token is still [CLS], the second still [CURRENT], and length
    # must fit in max_length.
    assert enc.input_ids[0] == tok.cls_id
    assert enc.input_ids[1] == tok.current_id
    assert len(enc.input_ids) == 8


def test_attention_mask_matches_non_pad_length():
    tok = _make_tokenizer(max_length=16)
    conv = Conversation.from_text("short")
    enc = tok.encode(conv)
    assert sum(enc.attention_mask) == enc.length
    # Padded positions
    for i, (a, tid) in enumerate(zip(enc.attention_mask, enc.input_ids)):
        if i >= enc.length:
            assert a == 0
            assert tid == tok.pad_id


def test_pooling_mask_never_covers_padding():
    tok = _make_tokenizer(max_length=16)
    conv = Conversation.from_text("hi")
    enc = tok.encode(conv)
    for i, p in enumerate(enc.pooling_mask):
        if p:
            assert enc.attention_mask[i] == 1


def test_batch_encoding_shapes_uniform():
    tok = _make_tokenizer(max_length=12)
    convs = [
        Conversation.from_text("short"),
        Conversation.from_text("a slightly longer message"),
    ]
    batch = tok.encode_batch(convs)
    assert len(batch["input_ids"]) == 2
    for key in ("input_ids", "attention_mask", "pooling_mask"):
        for row in batch[key]:
            assert len(row) == 12
