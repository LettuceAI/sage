"""The SAGE model: Jina v2 base encoder + role-aware tokenizer + current-span
mean-pool + linear multi-label head.

Param budget:
    ~137M encoder (Jina v2 base)
    +   ~3K role-token embeddings (4 new rows in the 768-dim table)
    +   ~5K classifier head (768 → 7 + bias)
    ≈ ~137M total.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sage.conversation import CURRENT_TOKEN, ROLE_TOKEN, Role
from sage.schema import NUM_CATEGORIES
from sage.tokenizer import DEFAULT_BASE_TOKENIZER, SageTokenizer

# Anchor words whose embeddings are averaged to initialize each new role
# token. Empirically-chosen semantic neighbours.
ROLE_INIT_ANCHORS: dict[str, list[str]] = {
    ROLE_TOKEN[Role.USER]: ["user", "human", "person"],
    ROLE_TOKEN[Role.CHAR]: ["character", "assistant", "bot"],
    ROLE_TOKEN[Role.SYSTEM]: ["system", "instruction", "prompt"],
    CURRENT_TOKEN: ["this", "now", "message"],
}


@dataclass
class SageConfig:
    base_model_name: str = "jinaai/jina-embeddings-v2-base-en"
    hidden_size: int = 768
    num_labels: int = NUM_CATEGORIES
    dropout: float = 0.2
    max_length: int = 1024


class SageModel(nn.Module):
    def __init__(self, config: SageConfig | None = None, *, encoder=None) -> None:
        super().__init__()
        self.config = config or SageConfig()

        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(self.config.base_model_name, trust_remote_code=True)
        self.encoder = encoder

        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pooling_mask: Tensor,
    ) -> Tensor:
        """Returns logits of shape ``(batch, num_labels)``. Sigmoid is applied
        by the loss (``BCEWithLogitsLoss``) at train time and by the inference
        wrapper at predict time."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)
        pooled = self._pool(hidden, pooling_mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    @staticmethod
    def _pool(hidden: Tensor, pooling_mask: Tensor) -> Tensor:
        """Masked mean pooling over the target-message span."""
        mask = pooling_mask.to(hidden.dtype).unsqueeze(-1)  # (B, L, 1)
        numer = (hidden * mask).sum(dim=1)  # (B, H)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return numer / denom

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_proba(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pooling_mask: Tensor,
    ) -> Tensor:
        return torch.sigmoid(self(input_ids, attention_mask, pooling_mask))


# ---------------------------------------------------------------------------
# Construction helper: builds tokenizer + model and wires role-token embeds
# ---------------------------------------------------------------------------
def build_model_and_tokenizer(
    base_model_name: str = DEFAULT_BASE_TOKENIZER,
    *,
    max_length: int = 1024,
    dropout: float = 0.2,
) -> tuple[SageModel, SageTokenizer]:
    tokenizer = SageTokenizer(base_tokenizer_name=base_model_name, max_length=max_length)
    config = SageConfig(base_model_name=base_model_name, max_length=max_length, dropout=dropout)
    model = SageModel(config=config)

    # Resize the encoder's token embedding to include the 4 new special tokens.
    if tokenizer.added_token_count > 0:
        model.encoder.resize_token_embeddings(tokenizer.vocab_size)
        _init_role_embeddings(model, tokenizer)

    return model, tokenizer


def _init_role_embeddings(model: SageModel, tokenizer: SageTokenizer) -> None:
    """Initialize the 4 new special-token embedding rows as the mean of the
    pretrained embeddings of semantic anchor words. Mutates ``model.encoder``
    in place. Falls back to a random normal if no anchors tokenize cleanly."""
    emb = model.encoder.get_input_embeddings()
    weight = emb.weight  # (V, H)
    for special, anchors in ROLE_INIT_ANCHORS.items():
        sp_id = tokenizer.tokenizer.convert_tokens_to_ids(special)
        anchor_vecs: list[Tensor] = []
        for anchor in anchors:
            pieces = tokenizer.tokenizer.encode(anchor, add_special_tokens=False)
            if not pieces:
                continue
            anchor_vecs.append(weight[pieces].mean(dim=0))
        if anchor_vecs:
            mean_init = torch.stack(anchor_vecs).mean(dim=0)
            with torch.no_grad():
                weight[sp_id].copy_(mean_init)
