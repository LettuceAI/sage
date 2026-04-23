"""Torch Dataset backed by SAGE JSONL; tokenization happens per item."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from sage.conversation import Conversation
from sage.schema import CATEGORIES, Category
from sage.tokenizer import SageTokenizer


class SageDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer: SageTokenizer) -> None:
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        conv = Conversation.from_dict(row["conversation"])
        encoded = self.tokenizer.encode(conv)
        labels_dict = row.get("labels") or {}
        labels = torch.tensor(
            [float(labels_dict.get(c.value, 0.0)) for c in CATEGORIES],
            dtype=torch.float32,
        )
        # Per-example observation mask. Missing "observed" key = fully observed
        # (back-compat with pre-masking JSONL files).
        observed_raw = row.get("observed")
        if observed_raw is None:
            obs_values = [1.0] * len(CATEGORIES)
        else:
            observed_set = set(observed_raw)
            obs_values = [1.0 if c.value in observed_set else 0.0 for c in CATEGORIES]
        observed_mask = torch.tensor(obs_values, dtype=torch.float32)
        return {
            "input_ids": torch.tensor(encoded.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoded.attention_mask, dtype=torch.long),
            "pooling_mask": torch.tensor(encoded.pooling_mask, dtype=torch.long),
            "labels": labels,
            "observed_mask": observed_mask,
        }

    def label_counts(self, threshold: float = 0.5) -> dict[Category, int]:
        counts: Counter[Category] = Counter()
        for row in self.rows:
            for k, v in (row.get("labels") or {}).items():
                if float(v) >= threshold:
                    try:
                        counts[Category(k)] += 1
                    except ValueError:
                        pass
        return dict(counts)

    def observed_counts(self) -> dict[Category, int]:
        """How many examples in the dataset explicitly observe each category.
        Used for ``pos_weight`` calibration — rarely-observed categories should
        normalize against their *observed* count, not the corpus total."""
        counts: Counter[Category] = Counter()
        for row in self.rows:
            observed_raw = row.get("observed")
            if observed_raw is None:
                for c in CATEGORIES:
                    counts[c] += 1
            else:
                for v in observed_raw:
                    try:
                        counts[Category(v)] += 1
                    except ValueError:
                        pass
        return dict(counts)


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """All examples are already padded to ``max_length`` by the tokenizer, so
    collation is a simple stack."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
