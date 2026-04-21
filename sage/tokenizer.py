"""Conversation → tokenizer input for SAGE.

Wraps a HuggingFace tokenizer (Jina v2's by default), adds the four SAGE
special tokens, and encodes a ``Conversation`` into:

- ``input_ids``      : token IDs
- ``attention_mask`` : 1 for real tokens, 0 for padding
- ``pooling_mask``   : 1 only for tokens belonging to the ``[CURRENT]`` span,
                       including the ``[CURRENT]`` marker token, excluding the
                       terminating ``[SEP]``. Used by the model to mean-pool
                       only over the target message.

Truncation follows the policy in ``docs/architecture.md`` §1.5:
1. Drop oldest context turns until the total fits.
2. If the target alone is too long, right-truncate it.
3. Keep ``[CLS]`` and the ``[CURRENT]`` marker unconditionally.
4. ``[SYSTEM]`` turn at position 0 is pinned unless keeping it forces the
   target to be truncated.
"""

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
    length: int  # number of non-pad tokens

    def validate(self) -> None:
        n = len(self.input_ids)
        assert len(self.attention_mask) == n, "attention_mask length mismatch"
        assert len(self.pooling_mask) == n, "pooling_mask length mismatch"
        assert any(self.pooling_mask), "pooling_mask must have at least one 1"
        # pooling_mask must be a subset of attention_mask
        for a, p in zip(self.attention_mask, self.pooling_mask, strict=True):
            if p and not a:
                raise AssertionError("pooling_mask includes a padding position")


class SageTokenizer:
    """Tokenizer wrapper that knows about SAGE's role tokens and pooling mask."""

    def __init__(
        self,
        base_tokenizer_name: str = DEFAULT_BASE_TOKENIZER,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        tokenizer=None,  # allow injection for tests
    ) -> None:
        self.max_length = max_length
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer  # local import keeps tests light

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_tokenizer_name, trust_remote_code=True
            )
        # Register SAGE special tokens once (idempotent — HF tokenizers skip existing ones).
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SPECIAL_TOKENS)}
        )
        self._added_token_count = added

        # Cache IDs we need at encode time.
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        assert self.cls_id is not None and self.sep_id is not None and self.pad_id is not None, (
            "base tokenizer must provide cls/sep/pad tokens"
        )

        def _single_id(tok: str) -> int:
            ids = self.tokenizer.convert_tokens_to_ids([tok])
            assert ids[0] != self.tokenizer.unk_token_id, f"special token {tok} not registered"
            return ids[0]

        self.role_id: dict[Role, int] = {r: _single_id(ROLE_TOKEN[r]) for r in Role}
        self.current_id: int = _single_id(CURRENT_TOKEN)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def added_token_count(self) -> int:
        """How many tokens this wrapper added on top of the base tokenizer.
        The model uses this to know whether to resize its embedding table."""
        return self._added_token_count

    def encode(self, conversation: Conversation) -> EncodedConversation:
        # 1. Tokenize each turn's text once (without special tokens).
        ctx_turn_ids: list[list[int]] = [self._encode_text(t.text) for t in conversation.context]
        target_text_ids = self._encode_text(conversation.target.text)

        # 2. Apply budget. Reserve room for: [CLS], per-turn ([role] ... [SEP]) wrappers.
        # Each context turn costs: 1 (role) + len(text) + 1 (sep).
        # Target turn costs:        1 ([CURRENT]) + len(text) + 1 (sep).
        # Plus [CLS] at the start.
        budget = self.max_length - 1  # leave 1 for [CLS]
        target_cost = 1 + len(target_text_ids) + 1
        # Truncate target if it's too big on its own.
        if target_cost > budget:
            max_target_text = budget - 2
            target_text_ids = target_text_ids[:max_target_text]
            target_cost = 1 + len(target_text_ids) + 1

        remaining = budget - target_cost
        # Determine if a pinned SYSTEM turn exists at position 0.
        has_pinned_system = conversation.context and conversation.context[0].role is Role.SYSTEM

        # Walk context from most-recent back to oldest, keeping turns that fit.
        # We'll reverse at the end to restore conversation order.
        kept_context_indices: list[int] = []
        cost_of_kept = 0
        # Reserve space for the pinned system turn (if any) so we don't drop it unless needed.
        reserved_system = 0
        if has_pinned_system:
            sys_cost = 1 + len(ctx_turn_ids[0]) + 1
            if sys_cost <= remaining:
                reserved_system = sys_cost

        # Iterate non-system context turns from newest to oldest.
        non_system_start = 1 if has_pinned_system else 0
        for i in range(len(conversation.context) - 1, non_system_start - 1, -1):
            turn_cost = 1 + len(ctx_turn_ids[i]) + 1
            if cost_of_kept + turn_cost + reserved_system <= remaining:
                kept_context_indices.append(i)
                cost_of_kept += turn_cost
            else:
                break
        kept_context_indices.reverse()
        # Prepend the pinned system turn if it fits.
        if has_pinned_system and reserved_system > 0:
            kept_context_indices = [0, *kept_context_indices]

        # 3. Assemble final token stream.
        input_ids: list[int] = [self.cls_id]
        pooling_mask: list[int] = [0]  # [CLS] is not pooled

        for i in kept_context_indices:
            turn = conversation.context[i]
            role_tok = self.role_id[turn.role]
            input_ids.append(role_tok)
            pooling_mask.append(0)
            input_ids.extend(ctx_turn_ids[i])
            pooling_mask.extend([0] * len(ctx_turn_ids[i]))
            input_ids.append(self.sep_id)
            pooling_mask.append(0)

        # Target turn: [CURRENT] ... text ... [SEP]
        input_ids.append(self.current_id)
        pooling_mask.append(1)  # the [CURRENT] marker counts toward pooling
        input_ids.extend(target_text_ids)
        pooling_mask.extend([1] * len(target_text_ids))
        input_ids.append(self.sep_id)
        pooling_mask.append(0)  # terminating [SEP] excluded

        attention_mask = [1] * len(input_ids)

        # 4. Pad to max_length.
        pad_needed = self.max_length - len(input_ids)
        if pad_needed > 0:
            input_ids.extend([self.pad_id] * pad_needed)
            attention_mask.extend([0] * pad_needed)
            pooling_mask.extend([0] * pad_needed)

        encoded = EncodedConversation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
            length=sum(attention_mask),
        )
        encoded.validate()
        return encoded

    def encode_batch(self, conversations: list[Conversation]) -> dict[str, list[list[int]]]:
        encoded = [self.encode(c) for c in conversations]
        return {
            "input_ids": [e.input_ids for e in encoded],
            "attention_mask": [e.attention_mask for e in encoded],
            "pooling_mask": [e.pooling_mask for e in encoded],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
