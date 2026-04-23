"""Export SAGE checkpoint to ONNX fp16 + INT8 with PyTorch parity check.

At seq=2048 the full fp32 graph exceeds protobuf's 2 GiB hard limit during
ONNX optimization (ALiBi bias + intermediate activations). Exporting the
model in fp16 halves every tensor and keeps the graph well inside the
limit while preserving accuracy (checked via parity against the fp32
PyTorch reference). INT8 quantization is then applied from the fp16
ONNX; the final INT8 file is ~35 MB regardless of the source dtype.
"""

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


def export_fp16(model: SageModel, tokenizer: SageTokenizer, out: Path, *, opset: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = _build_dummy_inputs(tokenizer)

    # Cast weights + buffers to fp16 so the ONNX graph fits under 2 GiB.
    # Input ids stay int64 — only float tensors in the model are cast.
    model_fp16 = model.half()

    torch.onnx.export(
        model_fp16,
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
    print(f"[export] fp16 ONNX → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def quantize_int8(fp16_path: Path, int8_path: Path) -> None:
    """Quantize fp16 ONNX → INT8.

    onnxruntime's dynamic quantizer expects fp32 input, so first convert the
    fp16 graph to fp32 in memory (weights up-cast), then quantize. Scratch
    fp32 file is kept in a tempdir so the final output is just the INT8.
    """
    import tempfile

    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        fp32_scratch = Path(tmp) / "model_fp32.onnx"
        model = onnx.load(str(fp16_path))

        # Up-cast every fp16 initializer to fp32.
        for init in model.graph.initializer:
            if init.data_type == onnx.TensorProto.FLOAT16:
                arr = numpy_helper.to_array(init).astype(np.float32)
                new = numpy_helper.from_array(arr, init.name)
                init.CopyFrom(new)

        # Flip any fp16 value-info / io tensor-types to fp32 so ORT accepts
        # the model for quantization.
        for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.FLOAT16:
                tt.elem_type = onnx.TensorProto.FLOAT

        # Any Cast-to-fp16 nodes still in the graph would produce wrong dtype
        # after initializer up-cast. Rewrite them to cast-to-fp32.
        for node in model.graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT16:
                        attr.i = onnx.TensorProto.FLOAT

        onnx.save(model, str(fp32_scratch))

        quantize_dynamic(
            model_input=str(fp32_scratch),
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

    # PyTorch reference — run in fp32 on the original (non-halved) weights.
    model.eval()
    with torch.inference_mode():
        ref = model(dummy["input_ids"], dummy["attention_mask"], dummy["pooling_mask"])
    ref_np = ref.cpu().float().numpy()

    ort_in = {k: v.numpy() for k, v in dummy.items()}
    out = sess.run(None, ort_in)[0].astype(np.float32)

    max_abs_diff = float(np.max(np.abs(ref_np - out)))
    print(f"[parity {label}] max|Δ| = {max_abs_diff:.5f}  (atol={atol})")
    if max_abs_diff > atol:
        raise SystemExit(f"parity check FAILED: max diff {max_abs_diff:.5f} > atol {atol}")
    print(f"[parity {label}] OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SAGE to ONNX (fp16) + INT8")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--base-model", default="jinaai/jina-embeddings-v2-base-en")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--out-fp16", type=Path, required=True)
    ap.add_argument("--out-int8", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--fp16-atol", type=float, default=5e-2)
    ap.add_argument("--int8-atol", type=float, default=0.2)
    args = ap.parse_args()

    # Load twice: one fp32 reference (for parity), one that we'll cast to fp16.
    model_ref, tokenizer = _load_checkpoint(
        args.checkpoint, args.base_model, args.max_length, args.dropout,
    )
    model_exp, _ = _load_checkpoint(
        args.checkpoint, args.base_model, args.max_length, args.dropout,
    )

    export_fp16(model_exp, tokenizer, args.out_fp16, opset=args.opset)
    parity_check(model_ref, tokenizer, args.out_fp16, atol=args.fp16_atol, label="fp16")

    if not args.skip_int8:
        quantize_int8(args.out_fp16, args.out_int8)
        parity_check(model_ref, tokenizer, args.out_int8, atol=args.int8_atol, label="int8")


if __name__ == "__main__":
    main()
