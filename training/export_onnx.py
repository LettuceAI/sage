"""Export SAGE checkpoint to ONNX fp32 + INT8 with PyTorch parity check."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sage.conversation import Conversation
from sage.model import SageModel, build_model_and_tokenizer
from sage.tokenizer import SageTokenizer

OPSET = 17
INPUT_NAMES = ["input_ids", "attention_mask", "pooling_mask"]
OUTPUT_NAMES = ["logits"]


def _load_checkpoint(
    checkpoint: Path, base_model: str, max_length: int, dropout: float
) -> tuple[SageModel, SageTokenizer]:
    model, tokenizer = build_model_and_tokenizer(
        base_model_name=base_model,
        max_length=max_length,
        dropout=dropout,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tokenizer


def _build_dummy_inputs(tokenizer: SageTokenizer, batch: int = 2) -> dict[str, torch.Tensor]:
    """Synthesize a small batch of encoded conversations for tracing."""
    convs = [Conversation.from_text(f"sample text {i}") for i in range(batch)]
    encoded = tokenizer.encode_batch(convs)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in encoded.items()}


def export_fp32(model: SageModel, tokenizer: SageTokenizer, out: Path, *, opset: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = _build_dummy_inputs(tokenizer)

    # Use the legacy tracer-based exporter (dynamo=False). The new dynamo
    # exporter produces bloated external-data files for custom-code encoders
    # like Jina v2 and is slower to run at our scale.
    torch.onnx.export(
        model,
        args=(dummy["input_ids"], dummy["attention_mask"], dummy["pooling_mask"]),
        f=str(out),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=opset,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "pooling_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] fp32 ONNX → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"[quant] INT8 ONNX → {int8_path}  ({int8_path.stat().st_size / 1e6:.1f} MB)")


def parity_check(
    model: SageModel,
    tokenizer: SageTokenizer,
    onnx_path: Path,
    *,
    atol: float,
    label: str,
) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = _build_dummy_inputs(tokenizer, batch=4)

    # PyTorch reference
    model.eval()
    with torch.inference_mode():
        ref = model(dummy["input_ids"], dummy["attention_mask"], dummy["pooling_mask"])
    ref_np = ref.cpu().numpy()

    # ONNX
    ort_in = {k: v.numpy() for k, v in dummy.items()}
    out = sess.run(None, ort_in)[0]

    max_abs_diff = float(np.max(np.abs(ref_np - out)))
    print(f"[parity {label}] max|Δ| = {max_abs_diff:.5f}  (atol={atol})")
    if max_abs_diff > atol:
        raise SystemExit(f"parity check FAILED: max diff {max_abs_diff:.5f} > atol {atol}")
    print(f"[parity {label}] OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SAGE to ONNX + INT8")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--base-model", default="jinaai/jina-embeddings-v2-base-en")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)  # inference: no dropout
    ap.add_argument("--out-fp32", type=Path, required=True)
    ap.add_argument("--out-int8", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--fp32-atol", type=float, default=1e-3)
    ap.add_argument("--int8-atol", type=float, default=0.2)
    args = ap.parse_args()

    model, tokenizer = _load_checkpoint(
        args.checkpoint,
        args.base_model,
        args.max_length,
        args.dropout,
    )

    export_fp32(model, tokenizer, args.out_fp32, opset=args.opset)
    parity_check(model, tokenizer, args.out_fp32, atol=args.fp32_atol, label="fp32")

    if not args.skip_int8:
        quantize_int8(args.out_fp32, args.out_int8)
        parity_check(model, tokenizer, args.out_int8, atol=args.int8_atol, label="int8")


if __name__ == "__main__":
    main()
