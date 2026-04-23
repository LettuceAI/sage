"""Role-aware tokenizer: Conversation -> (input_ids, attention_mask, pooling_mask)."""

from __future__ import annotations

from dataclasses import dataclass

from sage.conversation import (
    CURRENT_TOKEN,
    ROLE_TOKEN,
    SPECIAL_TOKENS,
    Conversation,
    Role,
)

DEFAULT_BASE_TOKENIZER = "jinaai/jina-embeddings-v2-base-en"
DEFAULT_MAX_LENGTH = 1024


@dataclass
class EncodedConversation:
    input_ids: list[int]
    attention_mask: list[int]
    pooling_mask: list[int]
    length: int

    def validate(self) -> None:
        n = len(self.input_ids)
        assert len(self.attention_mask) == n
        assert len(self.pooling_mask) == n
        assert any(self.pooling_mask)
        for a, p in zip(self.attention_mask, self.pooling_mask, strict=True):
            if p and not a:
                raise AssertionError("pooling_mask covers a pad position")


class SageTokenizer:
    def __init__(
        self,
        base_tokenizer_name: str = DEFAULT_BASE_TOKENIZER,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        tokenizer=None,
    ) -> None:
        self.max_length = max_length
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_tokenizer_name, trust_remote_code=True
            )
        self._added_token_count = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SPECIAL_TOKENS)}
        )

        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        assert self.cls_id is not None and self.sep_id is not None and self.pad_id is not None

        def _id(tok: str) -> int:
            ids = self.tokenizer.convert_tokens_to_ids([tok])
            assert ids[0] != self.tokenizer.unk_token_id, f"special token {tok} not registered"
            return ids[0]

        self.role_id: dict[Role, int] = {r: _id(ROLE_TOKEN[r]) for r in Role}
        self.current_id: int = _id(CURRENT_TOKEN)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def added_token_count(self) -> int:
        return self._added_token_count

    def encode(self, conversation: Conversation) -> EncodedConversation:
        ctx_ids: list[list[int]] = [self._encode_text(t.text) for t in conversation.context]
        target_ids = self._encode_text(conversation.target.text)

        budget = self.max_length - 1  # [CLS]
        target_cost = 1 + len(target_ids) + 1
        if target_cost > budget:
            target_ids = target_ids[: budget - 2]
            target_cost = 1 + len(target_ids) + 1

        remaining = budget - target_cost
        has_system = bool(conversation.context) and conversation.context[0].role is Role.SYSTEM
        reserved_system = 0
        if has_system:
            sys_cost = 1 + len(ctx_ids[0]) + 1
            if sys_cost <= remaining:
                reserved_system = sys_cost

        # Keep newest context turns, drop oldest first.
        start = 1 if has_system else 0
        kept: list[int] = []
        cost = 0
        for i in range(len(conversation.context) - 1, start - 1, -1):
            turn_cost = 1 + len(ctx_ids[i]) + 1
            if cost + turn_cost + reserved_system <= remaining:
                kept.append(i)
                cost += turn_cost
            else:
                break
        kept.reverse()
        if has_system and reserved_system > 0:
            kept = [0, *kept]

        input_ids: list[int] = [self.cls_id]
        pooling_mask: list[int] = [0]
        for i in kept:
            turn = conversation.context[i]
            input_ids.append(self.role_id[turn.role])
            pooling_mask.append(0)
            input_ids.extend(ctx_ids[i])
            pooling_mask.extend([0] * len(ctx_ids[i]))
            input_ids.append(self.sep_id)
            pooling_mask.append(0)

        input_ids.append(self.current_id)
        pooling_mask.append(1)
        input_ids.extend(target_ids)
        pooling_mask.extend([1] * len(target_ids))
        input_ids.append(self.sep_id)
        pooling_mask.append(0)

        attention_mask = [1] * len(input_ids)
        pad = self.max_length - len(input_ids)
        if pad > 0:
            input_ids.extend([self.pad_id] * pad)
            attention_mask.extend([0] * pad)
            pooling_mask.extend([0] * pad)

        enc = EncodedConversation(input_ids, attention_mask, pooling_mask, sum(attention_mask))
        enc.validate()
        return enc

    def encode_batch(self, conversations: list[Conversation]) -> dict[str, list[list[int]]]:
        encoded = [self.encode(c) for c in conversations]
        return {
            "input_ids": [e.input_ids for e in encoded],
            "attention_mask": [e.attention_mask for e in encoded],
            "pooling_mask": [e.pooling_mask for e in encoded],
        }

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
