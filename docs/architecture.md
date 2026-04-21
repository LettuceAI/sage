# SAGE architecture

SAGE is a **trajectory-aware** multi-label content moderation model. Unlike message-level classifiers, SAGE scores the **current message given its conversation history**, which is what enables detection of patterns that single-message classifiers miss: grooming escalation, coerced consent, emotional manipulation build-up, and self-harm ideation developing across a session.

- **Base encoder:** `jinaai/jina-embeddings-v2-base-en` (137M params, 12 layers, dim 768, ALiBi positions, **native 8192 context**)
- **Input:** a structured conversation with role markers, max 16 turns / 1024 tokens (headroom to 8192 in future versions without changing models)
- **Output:** 7 category scores, each ∈ [0, 1]
- **Size:** ~35 MB (INT8 dynamic quantization)
- **Latency:** ~30–50 ms on mid-range CPU at full 1024-token input (shorter inputs are proportionally faster)

---

## 1. Input format

### 1.1 Conceptual model

The model always classifies **exactly one message** — the *target message* — given zero or more preceding context turns.

```
context turns (0 to N-1)   →   [role_i] text_i [SEP] ...
target message (turn N)    →   [CURRENT] text_N
```

The `[CURRENT]` marker tells the model which message to score. Context turns influence the prediction but are not themselves scored.

### 1.2 Role tokens

Four special tokens are added to the tokenizer. They are single-token role markers, not textual prefixes.

| Token | Meaning |
|---|---|
| `[USER]` | A prior message from the end user |
| `[CHAR]` | A prior message from the AI character / assistant |
| `[SYSTEM]` | A system/persona prompt (usually only turn 0) |
| `[CURRENT]` | The message to classify (always the last turn) |

Rationale for four tokens rather than two (`[USER]`, `[CHAR]`):
- `[SYSTEM]` lets the model condition on persona information (important for roleplay where a persona can deliberately shift expected tone)
- `[CURRENT]` is architecturally critical — it's what the pooling layer keys on. Using a distinct token avoids any ambiguity when the last turn is also `[USER]`.

### 1.3 Token layout

A conversation of N turns becomes:

```
[CLS] [role_1] tok_1_1 tok_1_2 ... [SEP]
      [role_2] tok_2_1 tok_2_2 ... [SEP]
      ...
      [CURRENT] tok_N_1 tok_N_2 ... [SEP]
```

- `[CLS]` is kept but **not used for pooling** (see §3).
- `[SEP]` separates turns. We use the standard BERT `[SEP]` token, so no new sentinel is needed.
- Role tokens appear **before** each turn's text.

### 1.4 Length budget

| Parameter | Value |
|---|---|
| Max total tokens | 1024 |
| Max turns | 16 |
| Reserved for special tokens | ~35 (role + `[SEP]` × 16 + `[CLS]`) |
| Effective budget per turn (avg) | ~60 tokens |

The base model (`jina-embeddings-v2-base-en`) uses **ALiBi** (Attention with Linear Biases) position encoding and is pretrained on documents up to 8192 tokens. Running SAGE at 1024 tokens is well within its native range — no interpolation, retrofit, or position-embedding retraining is needed. The model will also extrapolate cleanly to longer contexts if v2 evaluation justifies raising the window.

### 1.5 Truncation policy

When the conversation exceeds 1024 tokens, turns are dropped from the **front** (oldest first), never the back. The `[CURRENT]` turn is never truncated — if a single message is itself > 800 tokens, it is truncated from its **end** (this is rare; long messages in roleplay tend to be worldbuilding, and the trailing portion usually carries the actual risk signal).

Truncation order:
1. Drop oldest context turns until total ≤ 1024
2. If `[CURRENT]` alone is still too long, right-truncate to fit
3. Always keep `[CLS]` and the `[CURRENT]` marker

A `[SYSTEM]` turn (if present and turn 0) is **pinned** — it is dropped only if keeping it would force truncation of `[CURRENT]`.

### 1.6 Worked example

Input (Python API):
```python
[
    {"role": "system", "text": "You are a friendly tutor helping a student."},
    {"role": "user", "text": "Hey, how was your day?"},
    {"role": "char", "text": "Good! Working through algebra with you."},
    {"role": "user", "text": "So what are your parents up to?"},
]
```

Token stream (conceptual):
```
[CLS] [SYSTEM] You are a friendly tutor ... [SEP]
      [USER] Hey , how was your day ? [SEP]
      [CHAR] Good ! Working through algebra with you . [SEP]
      [CURRENT] So what are your parents up to ? [SEP]
```

The model scores only the `[CURRENT]` turn ("So what are your parents up to?"). In isolation this is harmless; in a grooming escalation context it would score higher on the grooming axis because of what preceded it.

---

## 2. Single-message compatibility

A single message is a conversation of length 1:

```python
sage.check("some text")

# equivalent to:
sage.check([{"role": "user", "text": "some text"}])
```

Internally this becomes:
```
[CLS] [CURRENT] some text [SEP]
```

**Training covers both cases.** The base single-message datasets (Jigsaw, Civil Comments, etc.) are trained as 1-turn conversations so the model remains strong on in-isolation moderation, with trajectory-aware behavior emerging from the synthetic multi-turn training mix.

---

## 3. Pooling strategy

Standard MiniLM uses mean-pooling over all non-pad tokens. SAGE pools **only over tokens belonging to the `[CURRENT]` turn**, including the `[CURRENT]` marker itself, excluding `[CLS]` and context turns.

Mask construction:

```
pooling_mask[i] = 1   if  token i is in the span [CURRENT] ... [SEP]
                   0   otherwise (including [CLS], context turns, padding)

pooled = sum(hidden_states * pooling_mask) / sum(pooling_mask)
```

Why not `[CLS]` pooling: Jina v2 is trained with mean-pooling as its canonical sentence representation, not `[CLS]`. Restricting the mean to the `[CURRENT]` span is both faithful to the pretraining objective and more interpretable — each target-message token directly contributes.

Why not mean over all tokens: we want to classify *this message given context*, not "is this conversation bad on average." Pooling over context would dilute the signal and make the model output drift with long histories.

---

## 4. Classifier head

```
pooled (dim 768)
  → Dropout(0.2)
  → Linear(768 → 7)
  → [logits; sigmoid applied at inference]
```

One linear layer, 7 sigmoid outputs — one per category. Training uses `BCEWithLogitsLoss` (sigmoid fused into the loss for numerical stability); inference applies `sigmoid` explicitly.

Params: 768 × 7 + 7 = **5,383**. Negligible.

See [`sage/schema.py`](../sage/schema.py) for the category list and default thresholds.

---

## 5. Training

| Choice | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` + `pos_weight` per class; **focal (γ=2)** for `self_harm`, `grooming`, `sexual_minors` |
| Optimizer | AdamW, weight decay 0.01 |
| LR | Layer-wise decay: head `2e-4`, top layers `8e-5`, middle `3e-5`, bottom `1e-5` |
| Schedule | 10% warmup, cosine decay |
| Batch size | 32 (effective; grad accum if VRAM tight) |
| Epochs | 3–5, early stop on macro-F1 |
| Precision | fp16 mixed precision |
| Max seq len | 1024 (native range of Jina v2) |

### Embedding init for new role tokens

The four role tokens are new to the MiniLM vocabulary. Their embeddings are initialized from the mean of a small set of semantically related pretrained tokens:

| New token | Initialize from mean of |
|---|---|
| `[USER]` | "user", "human", "person" |
| `[CHAR]` | "character", "assistant", "bot" |
| `[SYSTEM]` | "system", "instruction", "role" |
| `[CURRENT]` | "this", "now", "message" |

Then fine-tuned end-to-end with the rest of the model.

---

## 6. Output format

```json
{
  "flagged": true,
  "categories": {
    "nsfw":          { "score": 0.04, "flagged": false },
    "violence":      { "score": 0.12, "flagged": false },
    "harassment":    { "score": 0.08, "flagged": false },
    "hate_speech":   { "score": 0.03, "flagged": false },
    "self_harm":     { "score": 0.02, "flagged": false },
    "grooming":      { "score": 0.71, "flagged": true  },
    "sexual_minors": { "score": 0.04, "flagged": false }
  }
}
```

`flagged` at the top level is `true` iff any category is flagged. Per-category thresholds are configurable at call time.

---

## 7. Design decisions — rejected alternatives

### Option: classify the entire conversation jointly
Rejected. Users want actionable per-message signals ("should I block *this* message?"). Joint classification is also harder to evaluate and harder to explain.

### Option: CLS-pooling
Rejected. MiniLM's CLS is not a trained sentence representation. Mean-pooling the target span is stronger and more interpretable.

### Option: concatenate all turns with a single `[USER]/[CHAR]` role token
Rejected. Without a distinct `[CURRENT]` marker, the model cannot unambiguously identify the message to classify when the last turn is `[USER]` (true in most cases).

### Option: cross-encoder over (context, current) pair
Rejected. Doubles the forward-pass cost and has no meaningful quality gain given the conversation is already short (≤ 512 tokens).

### Option: 2048+ token context
Deferred to v2. Jina v2 supports up to 8192 tokens natively, so going longer is a deployment decision, not an architectural one. 1024 / 16 turns is sufficient to catch realistic roleplay escalation patterns at an order of magnitude lower compute than 8192. Revisit if SAGE-Bench trajectory evaluations plateau against context length.

### Option: stay on MiniLM-L12 with position interpolation
Rejected. Interpolating a 512-context model to 1024 is well-tested but never as clean as native long context, and it bakes in a ceiling — we cannot go past ~1024 without further retraining. Jina v2's native 8192 context gives SAGE real headroom for future trajectory work without revisiting the base model.
