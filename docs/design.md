# Design

## What it is

SAGE is a multi-label content-moderation classifier for chat applications. It takes either a single message or up to 16 turns of conversation and returns a score in `[0, 1]` for each of seven harm categories: `nsfw`, `violence`, `harassment`, `hate_speech`, `self_harm`, `grooming`, `sexual_minors`.

The goal is a classifier that is:
- small enough to run on commodity hardware (target: ~35 MB INT8 artifact, ~30–50 ms CPU inference at 1024 tokens on a modern x86-64 core, single-threaded),
- accurate enough on standard benchmarks to be a drop-in replacement for hosted APIs (Perspective, OpenAI Moderation, Detoxify) in the categories it covers,
- context-aware — a message can be scored given the last N turns of a conversation rather than in isolation,
- deployable without any network dependency, so it fits privacy-sensitive, offline, embedded, and browser use cases.

## Why

Existing options fall into two clusters:

1. **Hosted APIs** (Perspective API, OpenAI Moderation, Azure AI Content Safety, etc.): free or cheap, but each message has to be sent to a third party, rate-limited, and subject to unilateral policy changes. Not acceptable for applications that promise local-only processing.
2. **Large safety models** (Llama Guard 3 8B, ShieldGemma 2B / 9B / 27B, WildGuard 7B): high quality, but 2B–8B parameters plus a generative decoding step. Too heavy for on-device inference and too slow for per-message checks on high-traffic services.

SAGE targets the middle: specialist quality on a small enough footprint that it can ship inside a Discord bot, a roleplay client, a browser extension, or a mobile app.

A secondary goal is *per-message in-conversation context* — the model encodes the last N turns so the classification of the current turn can reflect recent trajectory (e.g. escalating harassment, grooming patterns). This is a minor architectural addition rather than a separate research direction and is not yet exercised at training time; the stage-2 fine-tune (v1.1) is where synthetic multi-turn data is introduced.

## Technical approach

### Base model

`jinaai/jina-embeddings-v2-base-en` (Günther et al., 2023, [arXiv:2310.19923](https://arxiv.org/abs/2310.19923)). 137M parameters, 12 encoder layers, 768-dim hidden state, ALiBi positional bias (Press et al., 2021, [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)), 8192-token native context. Chosen over MiniLM-class encoders because:

- ALiBi's position-bias formulation extrapolates cleanly, so we get usable performance at any sequence length between training and 8192 without retraining the position table.
- Jina v2 is explicitly trained for long-context mean-pooled embeddings, which matches our pooling strategy.
- Size (137M) fits the INT8 artifact target (~35 MB) while remaining in the "BERT-family encoder" regime that standard fine-tuning techniques target.

### Added on top of the base

- Four special tokens — `[USER]`, `[CHAR]`, `[SYSTEM]`, `[CURRENT]` — added to the tokenizer. Embedding rows initialised from the mean of the pretrained embeddings of semantic anchor words (`user`, `human`, `person` for `[USER]`; etc.) then fine-tuned.
- A classifier head: `Linear(768 → 7)` with dropout 0.2. Sigmoid is applied by the loss (`BCEWithLogitsLoss`) at train time and by the inference wrapper at predict time.
- Mean pooling restricted to the `[CURRENT]` span of tokens. `[CLS]` is not used (MiniLM / Jina style mean-pooling is the pretrained signal).

Total added parameters: 4 × 768 + 768 × 7 + 7 ≈ 8,455 (~0.006 % of the model).

### Training

- Full fine-tuning. All 137M base parameters plus the added tokens and head are updated.
- **Layer-wise learning-rate decay** (discriminative fine-tuning, Howard & Ruder, 2018, [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)): head `2e-4`, encoder layers 10–11 `8e-5`, layers 6–9 `3e-5`, layers 0–5 and embeddings `1e-5`.
- **Loss**: per-class `BCEWithLogitsLoss` with `pos_weight` derived from each category's observed-negative-to-observed-positive ratio (clamped at 20). For `self_harm`, `grooming`, `sexual_minors` we apply focal reshaping (γ=2, Lin et al., 2017, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)) to down-weight confident correct predictions and concentrate gradient on hard cases.
- **Optimizer**: AdamW, weight decay 0.01, cosine decay with 10% linear warmup.
- **Precision**: fp16 mixed precision via CUDA AMP.

### Partial-label supervision

Each example carries an `observed: frozenset[Category]` listing the categories the source actually labels. During training the BCE loss is multiplied by a per-example 0/1 observation mask before reducing, so a source that only labels `hate_speech` (e.g. Measuring Hate Speech) contributes zero gradient to the other six heads. Without this, pooling heterogeneous datasets systematically poisons heads whose source didn't examine them. This is a standard partial-label / missing-label learning regime (see Durand et al., 2019, [arXiv:1902.06168](https://arxiv.org/abs/1902.06168) for a multi-label treatment). Per-source coverage is documented in [`DATA_LICENSES.md`](../DATA_LICENSES.md).

### Training data

Seven openly-licensed, non-gated sources. Approximate row counts after aggregation and deduplication (v1 pilot, `--limit-per-source 50000`):

| Source | Rows after dedup | Primary signal |
|---|---:|---|
| Civil Comments | ~44,000 | harassment, hate_speech, violence, nsfw |
| Measuring Hate Speech | ~26,000 | hate_speech only |
| Salad-Data | ~8,000 | broad taxonomy incl. grooming-adjacent |
| ProsocialDialog | ~44,000 | all-category negatives |
| HH-RLHF red-team | ~300 | adversarial harm prompts |
| NVIDIA Aegis | ~4,000 | includes dedicated `sexual_minors` label |
| WildChat-1M | ~40,000 | OpenAI Moderation-labelled per turn |
| **Total** | **~166,000** (before trajectorisation) | |

Positive-label distribution on the v1 train split:

| Category | Observed positives |
|---|---:|
| `hate_speech` | ~11,900 |
| `harassment` | ~5,000 |
| `violence` | ~2,600 |
| `nsfw` | ~1,800 |
| `self_harm` | ~440 |
| `grooming` | ~290 |
| `sexual_minors` | ~11 |

The three thinnest categories (`self_harm`, `grooming`, `sexual_minors`) drive the need for synthetic augmentation.

### Synthetic augmentation for rare categories

Public data for the three rare categories is insufficient to train stable per-class heads. Synthetic conversations are generated through an LLM with a mandatory human-review gate before merging into training data.

- Generation backend: llama-cpp-python loaded with a GBNF grammar compiled from the target JSON schema (`LlamaGrammar.from_json_schema`). The grammar constrains the sampler so the model cannot emit malformed structure — no invented field names, no wrong role values, no markdown fences. This eliminates the whole class of prompt-following bugs.
- Category policy: `sexual_minors` is negatives-only by hard rule; no synthetic positives are ever generated.
- Review: local Flask UI (`scripts/review_server.py`) with heuristic auto-flagging (near-duplicate openings, refusal artifacts, length anomalies, safety-gate terms) that reduces the reviewer's surface to the rejected subset.
- Storage: raw synthetic data is not released. Model weights trained on it are.

### Inference

Exported to ONNX (opset 17), INT8-quantised via `onnxruntime.quantization.quantize_dynamic`. A PyTorch-vs-ONNX parity check is run at fp32 (atol 1e-3) and INT8 (atol 0.2) before artifacts are released.

Long-message handling: if a single target message exceeds the available token budget after context reservation, the inference API splits the target into overlapping windows, classifies each, and max-aggregates per-category scores. This preserves full content coverage without raising the trained `max_length`. Chunking is transparent to the caller; `ModerationResult.n_chunks` records how many windows were used.

Rust inference is provided via `rust/sage-rs` (ort 2.0-rc), with a mirrored tokenizer and pooling implementation so both language runtimes produce identical scores.

## Evaluation

Two layers:

1. **Held-out validation metrics** on the standard train / val / test split (each example's observation mask is respected at eval time, so per-class F1 is computed only over observed positions). Reported: per-category precision / recall / F1, macro-F1 over categories with ≥1 observed validation example.
2. **Adversarial robustness** against `SAGE-Bench-Hard` — a repo-committed suite of ~1,500 curated hard positives and hard negatives tagged by failure mode (euphemism, obfuscation, sarcasm, negation, medical / academic / fiction / quotation framings, coded slang). The runner reports per-category P/R/F1 plus the 20 worst-performing failure modes. See [`benchmarks/sage_bench_hard/`](../benchmarks/sage_bench_hard).

At v1 release, cross-benchmark comparisons against Detoxify v0.5, Llama Guard 3 8B, and WildGuard 7B are planned on the categories shared with those systems (the toxicity / harassment / self-harm overlap). These numbers will be reported in the release notes, not asserted in advance.

### Current measured performance (epoch 1 pilot)

The v1 pilot trained through epoch 1 of 3 on a ~186k-row corpus at `max_length=256` on an RTX 4060 (constrained hardware for development). End-of-epoch-1 validation:

```
macro_f1 = 0.66   loss = 0.186

  nsfw            P=0.78  R=0.91  F1=0.84  n=126
  violence        P=0.62  R=0.93  F1=0.75  n=173
  harassment      P=0.41  R=0.95  F1=0.57  n=366
  hate_speech     P=0.64  R=0.94  F1=0.76  n=815
  self_harm       P=0.34  R=0.96  F1=0.50  n=22
  grooming        P=0.66  R=1.00  F1=0.79  n=21   (noisy — small n)
  sexual_minors   P=0.25  R=1.00  F1=0.40  n=1    (meaningless — n=1)
```

These numbers reflect a single epoch on a thin-pilot corpus without trajectory-aware synthesis. They are a sanity check, not a release claim. Final numbers will be reported after stage-2 fine-tuning on the synthesis-augmented corpus.

## Roadmap

### v1 — baseline

Training on the seven aggregated sources at `max_length=1024`, with the full synthesis-augmented rare-class data. Target: **macro-F1 ≥ 0.70** over the four well-covered categories, honest reporting on the thin three. Release: Apache 2.0 code, ONNX fp32 + INT8 artifacts, Python package, Rust crate.

### v1.1 — stage-2 fine-tune

Continued fine-tuning from `sage-v1/best.pt` on additional reviewed synthetic data with 10× lower LRs, 1–2 epochs. Primary target: raise `grooming` and `self_harm` F1 on SAGE-Bench-Hard. No architecture changes. The `--resume-from` flag in the training CLI supports this directly.

### v1.2 — optional context-length experiment

Retrain at `max_length=2048` if v1.1 eval shows context-length-bound failures on long chat histories. Would use FlashAttention-2 and gradient checkpointing to fit on 24 GB VRAM. Only executed if evaluation justifies the compute.

### v2 — prompt-injection / jailbreak detection

Add an eighth category for instruction override, persona injection, and system-prompt extraction. Separate data pipeline (JailbreakBench, deepset/prompt-injections, curated hard negatives from LettuceAI roleplay scenarios). Out of scope for v1.x.

## Non-goals

- **Ground-truth safety for every harm category.** SAGE emits scores. Threshold choice and action policy (log / warn / delete / timeout / block) belong to the calling application.
- **Replacement for large LLM judges.** A 137M classifier cannot match a 70B-class model on novel attack patterns, unseen harm categories, or compositional reasoning. SAGE targets the cases where those models are too expensive or too slow to run, not the cases where they excel.
- **Trajectory reasoning.** SAGE incorporates recent conversation context into per-message classification, but does not maintain session state, detect long-range patterns beyond its window, or reason about user intent. Application layers track rolling scores to detect session-level patterns SAGE alone cannot see in a single forward pass.
- **Multilingual support at v1.** Training data is English. Cross-lingual generalisation is incidental and not measured.

## Reproducibility

- Training is deterministic given `--seed` for data shuffling; ONNX export includes a parity check.
- 61 unit tests covering schema, tokenizer, pooling, loss, trajectory augmentation, synthesis, chunking, and inference coercion. Tests avoid downloading model weights — stub tokenizers are provided where needed.
- `make check` runs ruff, mypy, pytest, and `cargo check` end-to-end.
- GitHub Actions runs the full check across Python 3.10 / 3.11 / 3.12 plus the Rust crate on every PR.

## Open questions

- **Trajectory signal in practice.** The architecture supports multi-turn context, but stage-1 training data is overwhelmingly single-turn. Whether stage-2 synthetic trajectories are sufficient to teach context-dependent label flips will be determined at v1.1 evaluation.
- **Threshold calibration per deployment.** Default thresholds trade recall for precision on rare high-cost categories. Production deployments should calibrate via precision-recall curves on their own traffic before trusting the defaults.
- **Rare-class floor.** `sexual_minors` cannot be augmented with synthetic positives by policy. Its per-class F1 will remain noisy until more real labelled data becomes available; v1 will report this honestly rather than suppress the category.

## Licensing

Code is Apache 2.0. Model weights carry the union of constituent dataset licenses, summarised in [`DATA_LICENSES.md`](../DATA_LICENSES.md). Raw synthetic training data is not released.
