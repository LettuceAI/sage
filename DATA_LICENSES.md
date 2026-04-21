# Training data licenses

SAGE is trained on eight publicly available datasets. All sources are under **commercially-permissive licenses** and none require gated access. This document records license terms, attribution, and the per-source **observation coverage** that the training pipeline relies on for partial-signal handling.

The SAGE code is licensed under Apache 2.0. The SAGE model weights — because they are trained on datasets under mixed licenses — carry the **most restrictive obligations of the constituent datasets**, which in practice amounts to: attribution (CC-BY-4.0), attribution under ODC-BY, and inheritance of CC-BY-SA-3.0 on any redistributed underlying text. Redistributing the model weights is permitted; redistributing raw training data is not handled here.

---

## Observation coverage

Each source labels only a subset of SAGE's seven categories. The training pipeline records per-example observation masks so the classifier never learns a false "negative" for a category a source did not examine. The table below summarises each source's claimed coverage.

| Source | Observed categories |
|---|---|
| Jigsaw Toxic | harassment, violence, hate_speech |
| Civil Comments | harassment, violence, hate_speech, nsfw |
| Measuring Hate Speech | hate_speech |
| Salad-Data | all seven |
| ProsocialDialog (prosocial replies) | all seven (as negatives) |
| Anthropic HH-RLHF red-team | per-row, from tag |
| Aegis (Safe rows) | all seven |
| Aegis (harm rows) | the flagged subcategories only |
| WildChat-1M | all seven (OpenAI Moderation covers every category) |

---

## 1. Jigsaw Toxic Comment Classification Challenge

- **Source:** [huggingface.co/datasets/google/jigsaw_toxicity_pred](https://huggingface.co/datasets/google/jigsaw_toxicity_pred)
- **License:** CC0 1.0 (annotations), CC-BY-SA-3.0 (underlying Wikipedia talk-page text)
- **Commercial use:** Allowed
- **Attribution:** Wikipedia contributors; Jigsaw / Conversation AI
- **Categories used:** toxic, severe_toxic, obscene, threat, insult, identity_hate

## 2. Civil Comments

- **Source:** [huggingface.co/datasets/google/civil_comments](https://huggingface.co/datasets/google/civil_comments)
- **License:** CC0 1.0
- **Commercial use:** Allowed
- **Attribution:** Not required (CC0), but we credit Civil Comments / Google Jigsaw
- **Categories used:** toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit
- **Citation:** Borkan et al., 2019 — [arXiv:1903.04561](https://arxiv.org/abs/1903.04561)

## 3. Measuring Hate Speech

- **Source:** [huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
- **License:** CC-BY-4.0
- **Commercial use:** Allowed (with attribution)
- **Attribution required:** D-Lab, UC Berkeley
- **Citation:**
  ```
  Kennedy, C.J., Bacon, G., Sahn, A., & von Vacano, C. (2020).
  Constructing interval variables via faceted Rasch measurement and multitask deep learning:
  a hate speech application. arXiv:2009.10277.
  ```

## 4. Salad-Data

- **Source:** [huggingface.co/datasets/OpenSafetyLab/Salad-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data)
- **License:** Apache 2.0
- **Commercial use:** Allowed
- **Attribution:** OpenSafetyLab
- **Categories used:** Adult Content, Hate Speech & Discrimination, Child Abuse, Illegal activity subsets

## 5. ProsocialDialog

- **Source:** [huggingface.co/datasets/allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog)
- **License:** CC-BY-4.0
- **Commercial use:** Allowed (with attribution)
- **Attribution required:** Allen Institute for AI
- **Role in SAGE:** Primary source of **hard negatives** — prosocial responses to difficult prompts teach the model what is *not* a violation.
- **Citation:**
  ```
  Kim, H., Yu, Y., Jiang, L., Lu, X., Khashabi, D., Kim, G., Choi, Y., & Sap, M. (2022).
  ProsocialDialog: A Prosocial Backbone for Conversational Agents. EMNLP 2022.
  ```

## 6. Anthropic HH-RLHF (red-team subset)

- **Source:** [huggingface.co/datasets/Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **License:** MIT
- **Commercial use:** Allowed
- **Attribution:** Anthropic
- **Subset used:** `red-team-attempts/` only — diverse adversarial prompts across harm categories
- **Note:** The upstream authors advise against training **dialogue agents** on the harmlessness preference data. SAGE is a **classifier**, not a dialogue agent; the red-team prompts are used only as labeled harmful examples for supervised classification.

## 7. NVIDIA Aegis Content Safety Dataset

- **Source:** [huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0)
- **License:** CC-BY-4.0
- **Commercial use:** ✅ Allowed (with attribution)
- **Gated:** No
- **Attribution required:** NVIDIA
- **Size:** ~12k labeled prompts/responses across 13 harm subcategories
- **Role in SAGE:** Provides reliable cross-category negatives from "Safe" rows and subcategory-labelled positives. Uniquely includes a dedicated ``Sexual Minor`` label that fills the largest gap in SAGE's taxonomy.
- **Citation:**
  ```
  Ghosh, S., Varshney, P., Galinkin, E., & Parisien, C. (2024).
  AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts.
  ```

## 8. WildChat-1M (AI2)

- **Source:** [huggingface.co/datasets/allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
- **License:** ODC-BY (Open Data Commons Attribution)
- **Commercial use:** ✅ Allowed (with attribution)
- **Gated:** No (the ``WildChat-1M`` cleaned release is public; ``WildChat-1M-Full`` is the gated variant — we do not use it)
- **Attribution required:** Allen Institute for AI
- **Size:** ~838k real LLM conversations with per-turn OpenAI Moderation + Detoxify scores
- **Role in SAGE:** Scale and realism. Natural multi-turn chat data with OpenAI Moderation labels that cover every SAGE category 1:1. User turns are emitted as single-turn Examples; future iterations may emit full trajectories.
- **Citation:**
  ```
  Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., & Deng, Y. (2024).
  WildChat: 1M ChatGPT Interaction Logs in the Wild. ICLR 2024.
  ```

---

## Datasets explicitly excluded

These were evaluated and rejected for SAGE v1:

| Dataset | License | Reason |
|---|---|---|
| PKU-Alignment/BeaverTails | CC-BY-NC-4.0 | Non-commercial only |
| lmsys/toxic-chat | CC-BY-NC-4.0 | Non-commercial only |
| allenai/wildguardmix | ODC-BY (gated) | Gated behind AI2 Responsible Use acceptance; replaced by Aegis + WildChat |
| microsoft/toxigen | Access-form gated | Form-gated, license terms unclear for commercial use |

---

## Synthetic augmentation

Rare categories (self_harm, grooming) are augmented with synthetic examples generated by larger LLMs and human-reviewed before inclusion. Synthetic examples are generated and owned by LettuceAI and released under Apache 2.0 alongside the SAGE model.

**`sexual_minors` is a zero-tolerance category.** We do **not** generate synthetic positive examples for it. Positive signal comes only from the Salad-Data child-safety subset. Synthetic augmentation for this category is limited to **hard negatives** (e.g. non-sexual references to minors) to reduce false positives.
