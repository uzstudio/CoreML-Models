#!/usr/bin/env python3
"""
Convert FastSAM (CASIA-IVA-Lab/FastSAM) to Core ML.

FastSAM is *not* a SAM-style encoder/prompt-decoder model — it is a YOLOv8-seg
instance-segmentation network trained on a 2% subset of SA-1B. It segments every
object in one forward pass; point / box / text prompts are applied afterwards in
post-processing by selecting among the predicted instances. Architecturally it is
identical to the YOLO-seg detectors already in this repo (see YOLOEDemo), so the
conversion mirrors `conversion_scripts/convert_*` / SamKit's YOLO-World converter.

The exported model has a single image input and four split outputs so the Swift
side can assemble masks generically:

  image       [1, 3, 640, 640]  float32, RGB, scaled to [0, 1] (letterboxed)
  ->
  boxes       [1, 4, 8400]      cx, cy, w, h in 640-px (input) coordinates
  scores      [1, 1, 8400]      single "object" class, sigmoid-calibrated
  mask_coeffs [1, 32, 8400]     per-anchor mask coefficients
  mask_protos [1, 32, 160, 160] mask prototypes (instance mask = sigmoid(coeffs . protos))

Variants:
  FastSAM-s  (YOLOv8s-seg backbone, ~23 MB CoreML FP16) — on-device default
  FastSAM-x  (YOLOv8x-seg backbone, ~138 MB CoreML FP16) — best quality

Usage:
    pip install ultralytics coremltools torch
    python convert_fastsam.py                 # converts both s and x
    python convert_fastsam.py --size s        # just FastSAM-s
    python convert_fastsam.py --size x --verify
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb


# ---------------------------------------------------------------------------
# coremltools patch: `int` op for scalar shape casts (same fix used in YOLOEDemo).
# YOLOv8's Detect head casts dynamic shapes to int; coremltools 7+/9 trips over
# multi-dim casts. Force a constant when the value is known, else a plain cast.
# ---------------------------------------------------------------------------
def _patched_int(context, node):
    inputs = _ct_ops._get_inputs(context, node)
    x = inputs[0]
    if x.val is not None:
        val = x.val
        if isinstance(val, np.ndarray):
            val = int(val.item()) if val.ndim == 0 else int(val.flat[0])
        else:
            val = int(val)
        res = mb.const(val=np.int32(val), name=node.name)
    else:
        res = mb.cast(x=x, dtype="int32", name=node.name)
    context.add(res)


_ct_ops._TORCH_OPS_REGISTRY.register_func(_patched_int, torch_alias=["int"], override=True)


# FastSAM checkpoints (auto-downloaded from the Ultralytics asset mirror).
FASTSAM_MODELS = {
    "s": "FastSAM-s.pt",   # YOLOv8s-seg backbone
    "x": "FastSAM-x.pt",   # YOLOv8x-seg backbone
}
FASTSAM_DESCRIPTIONS = {
    "s": "FastSAM-s (YOLOv8s-seg)",
    "x": "FastSAM-x (YOLOv8x-seg)",
}

NUM_MASK_COEFFS = 32   # YOLOv8-seg mask prototypes / coefficients


# ---------------------------------------------------------------------------
# Wrapper: expose split boxes / scores / coeffs / protos
# ---------------------------------------------------------------------------
class FastSAMWrapper(nn.Module):
    """Wraps the YOLOv8-seg model so tracing yields four clean tensors.

    Ultralytics' Segment head, in inference (non-export) mode, returns
        out[0] = cat([decoded_boxes_cls, mask_coeffs], dim=1)  # [1, 4+nc+32, 8400]
        out[1] = (raw_feats, mask_coeffs, protos)              # protos = out[1][-1]
    Boxes come out already decoded to xywh in input-image (640) pixels and the
    class score is already sigmoid-activated, exactly like the YOLO-World port.
    """

    def __init__(self, seg_model: nn.Module, num_classes: int):
        super().__init__()
        self.model = seg_model
        self.nc = num_classes

    def forward(self, image):
        out = self.model(image)
        # ultralytics 8.4: out[0] == (pred [1,4+nc+32,8400], protos [1,32,160,160]);
        # older builds returned pred at out[0] and protos as the last item of out[1].
        first = out[0]
        if isinstance(first, (tuple, list)):
            pred, protos = first[0], first[1]
        else:
            pred = first
            aux = out[1]
            protos = aux["proto"] if isinstance(aux, dict) else aux[-1]

        boxes = pred[:, :4, :]                          # [1, 4, 8400]   decoded xywh (640 px)
        scores = pred[:, 4:4 + self.nc, :]              # [1, nc, 8400]  sigmoid score
        mask_coeffs = pred[:, 4 + self.nc:, :]          # [1, 32, 8400]
        return boxes, scores, mask_coeffs, protos


def _build_wrapper(checkpoint: str):
    """Load a FastSAM checkpoint and return (wrapper, num_classes)."""
    from ultralytics import FastSAM

    fs = FastSAM(checkpoint)
    seg_model = fs.model            # underlying YOLOv8-seg nn.Module
    seg_model.eval()

    # Determine class count from the Segment head (FastSAM uses a single class).
    head = seg_model.model[-1]
    num_classes = int(getattr(head, "nc", 1))

    wrapper = FastSAMWrapper(seg_model, num_classes)
    wrapper.eval()
    return wrapper, num_classes


def _image_input(input_size: int):
    """ImageType input: CoreML resizes + normalises on ANE/GPU; Swift feeds a CVPixelBuffer."""
    try:
        layout = ct.colorlayout.RGB
    except AttributeError:
        layout = "RGB"
    return ct.ImageType(
        name="image",
        shape=(1, 3, input_size, input_size),
        scale=1.0 / 255.0,
        color_layout=layout,
    )


def convert(size_key: str, output_dir: Path, input_size: int = 640,
            verify: bool = False, tensor_input: bool = False):
    checkpoint = FASTSAM_MODELS[size_key]
    desc = FASTSAM_DESCRIPTIONS[size_key]
    input_kind = "TensorType" if tensor_input else "ImageType (CVPixelBuffer, /255 baked in)"
    print(f"=== Converting {desc} @ {input_size}x{input_size}  [{input_kind}] ===")

    wrapper, nc = _build_wrapper(checkpoint)
    print(f"  classes={nc} mask_coeffs={NUM_MASK_COEFFS}")

    dummy = torch.randn(1, 3, input_size, input_size)

    # Warm up so lazily-built anchor/stride buffers are materialised before trace.
    with torch.no_grad():
        _ = wrapper(dummy)
        ref = wrapper(dummy)

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, check_trace=False)

    print("  Converting to CoreML (FP16)...")
    if tensor_input:
        inputs = [ct.TensorType(name="image", shape=(1, 3, input_size, input_size))]
    else:
        inputs = [_image_input(input_size)]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores"),
            ct.TensorType(name="mask_coeffs"),
            ct.TensorType(name="mask_protos"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.author = "CoreML-Models"
    mlmodel.short_description = f"{desc} @ {input_size} — promptable everything-segmentation"
    mlmodel.version = "1.0.0"

    out_path = output_dir / f"FastSAM_{size_key}_{input_size}.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  Saved {out_path}")

    if verify and tensor_input:   # ImageType verify needs a CVPixelBuffer, skip on raw tensor
        _verify(mlmodel, ref, dummy)

    return out_path


def _verify(mlmodel, torch_ref, dummy):
    """Compare CoreML outputs against the traced PyTorch reference."""
    print("  Verifying parity vs PyTorch...")
    names = ["boxes", "scores", "mask_coeffs", "mask_protos"]
    pred = mlmodel.predict({"image": dummy.numpy()})
    for name, ref in zip(names, torch_ref):
        cm = np.asarray(pred[name])
        diff = np.abs(cm.reshape(-1) - ref.detach().numpy().reshape(-1)).max()
        print(f"    {name:12s} max|Δ| = {diff:.4e}")


def main():
    parser = argparse.ArgumentParser(description="Convert FastSAM to Core ML")
    parser.add_argument("--size", choices=["s", "x", "both"], default="s",
                        help="FastSAM backbone (default: s)")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640],
                        help="Square input size(s); pass multiple to produce a set, "
                             "e.g. --input-size 320 512 640 (default: 640)")
    parser.add_argument("--output", default="../converted_models/FastSAM",
                        help="Output directory for the .mlpackage(s)")
    parser.add_argument("--tensor-input", action="store_true",
                        help="Use TensorType (MLMultiArray) input instead of the default "
                             "ImageType (CVPixelBuffer). Slower on-device.")
    parser.add_argument("--verify", action="store_true",
                        help="Check CoreML output parity against PyTorch (TensorType only)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = ["s", "x"] if args.size == "both" else [args.size]
    for size_key in sizes:
        for input_size in args.input_size:
            convert(size_key, output_dir, input_size, args.verify, args.tensor_input)

    print("\nDone. Add the .mlpackage(s) to the demo app, or distribute via Releases.")


if __name__ == "__main__":
    main()
