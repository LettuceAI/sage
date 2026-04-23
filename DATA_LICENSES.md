# Training data licenses

Seven sources. All permit commercial use and public model release. Attribution obligations carry through to released model weights.

## Observation coverage

Per-example `observed_mask` records which categories a source actually labels. The trainer masks loss on unobserved positions to prevent partial-coverage sources from contaminating other heads.

| Source | Observed categories |
|---|---|
| Civil Comments | harassment, violence, hate_speech, nsfw |
| Measuring Hate Speech | hate_speech |
| Salad-Data | all seven |
| ProsocialDialog | all seven (as negatives) |
| Anthropic HH-RLHF red-team | per row, from tag |
| NVIDIA Aegis (Safe rows) | all seven |
| NVIDIA Aegis (harm rows) | flagged subcategory only |
| WildChat-1M | all seven (via OpenAI Moderation scores) |

## Sources

### Civil Comments
- `google/civil_comments` — CC0 1.0
- 1.8M comments with toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit
- Borkan et al., 2019 — [arXiv:1903.04561](https://arxiv.org/abs/1903.04561)

### Measuring Hate Speech
- `ucberkeley-dlab/measuring-hate-speech` — CC-BY-4.0
- Kennedy et al., 2020 — [arXiv:2009.10277](https://arxiv.org/abs/2009.10277)

### Salad-Data
- `OpenSafetyLab/Salad-Data` — Apache 2.0
- Hierarchical taxonomy across sexual, violence, hate, self-harm, child abuse, and others

### ProsocialDialog
- `allenai/prosocial-dialog` — CC-BY-4.0
- Prosocial responses used as all-category negatives
- Kim et al., 2022 — EMNLP

### Anthropic HH-RLHF (red-team subset)
- `Anthropic/hh-rlhf`, `red-team-attempts/` only — MIT
- Tag-based labels; rows without mapped tags are dropped

### NVIDIA Aegis
- `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` — CC-BY-4.0
- Includes a dedicated `Sexual Minor` label
- Ghosh et al., 2024

### WildChat-1M
- `allenai/WildChat-1M` — ODC-BY (cleaned release is not gated)
- Per-turn OpenAI Moderation and Detoxify scores
- Zhao et al., 2024 — ICLR

## Excluded

| Dataset | Reason |
|---|---|
| `PKU-Alignment/BeaverTails` | CC-BY-NC-4.0 (non-commercial) |
| `lmsys/toxic-chat` | CC-BY-NC-4.0 (non-commercial) |
| `allenai/wildguardmix` | Gated behind Responsible Use Guidelines |
| `microsoft/toxigen` | Form-gated, unclear commercial terms |
| `google/jigsaw_toxicity_pred` | Script-based loader, unsupported by modern `datasets` |

## Synthetic augmentation

Rare categories (`grooming`, `self_harm`) are augmented with LLM-generated conversations that pass through a human-review gate before merging. `sexual_minors` is augmented with negatives only — no synthetic positives by policy. Raw synthetic data is not released; model weights trained on it are Apache 2.0.

## Effective obligations on model weights

Most restrictive constituent license is CC-BY-SA (via Wikipedia attribution in Civil Comments' upstream). In practice: attribution to the constituent datasets, patent grant under Apache 2.0 on the code.
