# Design

## What it is

SAGE is a multi-label content-moderation classifier for chat applications. It takes either a single message or up to 16 turns of conversation and returns a score in `[0, 1]` for each of seven harm categories: `nsfw`, `violence`, `harassment`, `hate_speech`, `self_harm`, `grooming`, `sexual_minors`.

The goal is a classifier that is:
- small enough to run on commodity hardware (target: ~35 MB INT8 artifact, ~30–50 ms CPU inference at 1024 tokens),
- accurate enough on standard benchmarks to be a drop-in replacement for hosted APIs (Perspective, OpenAI Moderation, Detoxify) in the categories it covers,
- context-aware, in the sense that a message can be scored given the last few turns of a conversation rather than in isolation,
- deployable without any network dependency, so it fits privacy-sensitive, offline, embedded, and browser use cases.

## Why

Existing options fall into two clusters:

1. **Hosted APIs** (Perspective, OpenAI Moderation, etc.): free or cheap, but each message has to be sent to a third party, rate-limited, and subject to unilateral policy changes. Not acceptable for applications that promise local-only processing.
2. **Large safety models** (Llama Guard, ShieldGemma, WildGuard): high quality, but 2B–8B parameters. Too heavy for on-device inference and too slow for per-message checks on high-traffic services.

SAGE is aimed at the middle: specialist quality on a small enough footprint that it can ship inside a Discord bot, a roleplay client, a browser extension, or a mobile app.

The secondary goal is to support *per-message in-conversation context* — the model encodes the last N turns so the classification of the current turn can reflect recent trajectory (e.g. escalating harassment, grooming patterns). This is a minor architectural addition rather than a separate research direction; it has not been exercised at training time yet.

## Technical approach

### Base model

`jinaai/jina-embeddings-v2-base-en`. 137M parameters, 12 encoder layers, 768-dim hidden state, ALiBi positional bias, 8192 tokens native context. Chosen over MiniLM-class encoders because the 8192-native context removes the need for position-embedding interpolation when we want longer windows, and because ALiBi extrapolates cleanly to any length at inference.

### Added on top

- Four new special tokens — `[USER]`, `[CHAR]`, `[SYSTEM]`, `[CURRENT]` — with embeddings initialized from the mean of semantic anchor words, then fine-tuned end-to-end.
- A classifier head: `Linear(768 → 7)` with dropout, sigmoid applied at inference.
- Mean pooling restricted to the `[CURRENT]` span (the target message), not `[CLS]` and not the full sequence.

### Training

- Full fine-tuning with layer-wise learning-rate decay (head `2e-4`, top layers `8e-5`, middle `3e-5`, bottom `1e-5`).
- `BCEWithLogitsLoss` with per-class `pos_weight` derived from observed counts; focal reshaping (γ=2) on `self_harm`, `grooming`, `sexual_minors` to handle their low positive rates.
- fp16 mixed precision, cosine decay with 10% warmup, AdamW.

### Partial-label supervision

Each example carries an `observed` set listing the categories the source actually labels. The trainer multiplies BCE loss by a 0/1 observation mask, so a source that only labels `hate_speech` (Measuring Hate Speech, for example) contributes zero gradient to the other six heads. Without this, pooling heterogeneous datasets produces systematic false negatives — a message that is NSFW but not hate-speech would effectively label NSFW=0 whenever it came from MHS. Per-source coverage is documented in `DATA_LICENSES.md`.

### Synthetic augmentation for rare categories

Public datasets provide very thin positive signal for `grooming` (~300 rows across Salad-Data after filtering), `self_harm` (~400), and essentially zero for `sexual_minors`. We generate synthetic conversations through an LLM with a human-review gate before merging into training data. Synthetic positives are gated by category policy: `sexual_minors` is negatives-only by hard rule.

The synthesis uses llama.cpp's GBNF grammar feature to constrain generator output to the exact JSON schema we expect, which eliminates prompt-following failures (invented field names, wrong role values, markdown fences). Review happens through a local Flask UI that can auto-flag near-duplicates, refusal artifacts, length anomalies, and unsafe patterns, leaving the reviewer to verify a small filtered subset.

### Inference

Exported to ONNX (opset 17), INT8-quantized via `onnxruntime.quantization.quantize_dynamic`. Parity-checked against PyTorch at fp32 and INT8 before release.

Long-message handling: if a single target message exceeds the sequence budget, the tokenizer splits it into overlapping windows and the inference API max-aggregates per-category scores across windows. This keeps `max_length` fixed without truncating content.

## Roadmap

### v1 — baseline (in progress)

- Training on the seven aggregated sources at `max_length=1024`, synthesis augmentation merged.
- Target macro-F1 on the validation split: **≥ 0.70** excluding `sexual_minors` (which remains marginal until positive data improves).
- Release: Apache 2.0 code, ONNX fp32 + INT8 artifacts, Python package, Rust crate.

### v1.1 — stage-2 fine-tune

Continued fine-tuning from `sage-v1/best.pt` on additional reviewed synthetic data with 10× lower LRs, 1–2 epochs. Primary target: lift `grooming` and `self_harm` F1 on SAGE-Bench-Hard from the v1 baseline. No architecture changes.

### v1.2 — context-length experiment

Optional retrain at `max_length=2048` if v1.1 eval shows significant failures on long chat histories. Would use FlashAttention-2 and gradient checkpointing to fit on 24 GB VRAM.

### v2 — prompt-injection / jailbreak detection

Add an eighth category for prompt-injection attempts (instruction override, persona injection, system-prompt extraction). Separate data pipeline; trained alongside the existing seven. Out of scope for v1.x.

## Evaluation

Two layers:

1. **Standard metrics** on the held-out validation set: per-category precision / recall / F1, macro-F1, per-failure-mode error rates via `SAGE-Bench-Hard`.
2. **Adversarial robustness** against `SAGE-Bench-Hard`, an in-repo suite of ~1,500 curated hard positives and hard negatives (euphemism, obfuscation, sarcasm, negation, medical/academic/fiction/quotation framings, coded slang). Scored as part of every release.

Cross-benchmark comparisons against Detoxify, WildGuard, and Llama Guard will be reported at v1 release on the categories they share with SAGE.

## Non-goals

- **Ground-truth safety for every harm category.** SAGE is a signal, not a policy engine. Thresholds and action policies belong to the calling application.
- **Replacement for large LLM judges.** A 137M classifier cannot match a 70B-class model on novel attack patterns, unseen harm categories, or compositional reasoning. SAGE targets the cases where those models are too expensive or too slow to run.
- **Trajectory *reasoning*.** SAGE incorporates recent conversation context into per-message classification, but does not maintain session state, detect long-range patterns beyond its window, or reason about user intent. Application layers can track rolling scores to detect patterns SAGE alone can't see in a single forward pass.
- **Multilingual support at v1.** Training data is English. Cross-lingual generalization is incidental and not measured.

## Open questions

- **Trajectory signal in practice.** The architecture supports multi-turn context, but stage-1 training data is overwhelmingly single-turn. Whether v1.1 synthetic trajectories are sufficient to teach context-dependent label flips (innocuous message in an escalation chain) will be determined at evaluation.
- **Calibration per deployment.** Default thresholds are set to trade recall for precision (lower thresholds on rare high-cost categories). Real-world deployments will likely want to tune these per-category based on their own false-positive tolerance.
- **Rare-class floor.** `sexual_minors` cannot be trained on synthetic positives by policy. The category will remain dependent on the Aegis subset and other cleanly-labeled sources. Below ~1000 positives, per-class F1 is dominated by noise.

## Licensing

Code is Apache 2.0. Model weights carry the union of constituent dataset licenses, summarized in `DATA_LICENSES.md`. Raw synthetic training data is not released.
