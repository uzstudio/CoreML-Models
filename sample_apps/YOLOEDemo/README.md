# YOLOEDemo

Open-vocabulary detection **and** instance segmentation on iOS using [YOLOE](https://github.com/THU-MIG/yoloe) + MobileCLIP.

Type any text — "person", "forklift", "coffee cup" — and detect/segment it in real-time camera, photos, or videos. No fixed class list.

## Architecture — region embeddings, similarity in Swift

Unlike YOLOWorldDemo (which bakes the text prompt into the class head and outputs
finished `scores`), YOLOEDemo keeps the open-vocabulary property fully decoupled:
the detector emits the **per-anchor region embedding** *before* the class logits,
and the region–text similarity is computed in Swift against cached text
embeddings. The image branch never sees the text, so **changing the query never
re-runs the detector** — only a cheap matmul.

```
Text query ─→ MobileCLIP text encoder ─→ reprta (RepRTA MLP) ─→ L2-norm, append 1.0
                                                                      │
                                                            text'  [N, 513]   (cached in Swift)
                                                                      │  similarity (BLAS)
Camera/Image ─→ YOLOE detector ─→ region_embeddings [1, 513, 8400] ───┤  logit = text' · region'
                              ├─→ boxes        [1, 4,  8400]           │  score = sigmoid(logit)
                              ├─→ mask_coeffs  [1, 32, 8400]           │
                              └─→ mask_protos  [1, 32, 160, 160]       └─→ NMS → boxes + masks
```

### Why 513 dimensions?

YOLOE's class head is a per-scale `BNContrastiveHead`:

```
logit = (BN(region) · normalize(text)) · exp(logit_scale) + bias
```

The exported detector folds the per-scale `exp(logit_scale)` into the first 512
embedding channels and appends the per-scale `bias` as channel 512. The cached
text gets a trailing constant `1.0`. The dot product then reproduces the exact
logit with a single matmul and **zero knowledge of the anchor layout** in Swift.
Verified parity vs the PyTorch head: `max |sigmoid(text'·region') − scores| ≈ 2e-11`,
and through the FP16 CoreML model the top detection's anchor/class match exactly.

## Models

| Model | Size | Description |
|-------|------|-------------|
| `yoloe_detector.mlpackage` | ~20 MB (FP16) | YOLOE-11s-seg region-embedding detector + seg |
| `reprta.mlpackage` | ~6 MB (FP32) | YOLOE RepRTA text-refinement MLP (`raw_tpe → tpe`) |
| `mobileclip_blt_text.mlpackage` | ~121 MB | Apple MobileCLIP B-LT text encoder (`text → final_emb_1`) |
| `clip_vocab.json` | 1.6 MB | CLIP BPE vocabulary for the Swift tokenizer |

`mobileclip_blt_text.mlpackage` is Apple's official Core ML export from
[apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) and is used as-is.

## Features

- **Camera**: real-time open-vocabulary detection
- **Photo**: pick from library, detect + colored instance masks, re-threshold without re-running
- **Video**: pick a video, detect frame-by-frame with overlay
- **Open-vocabulary**: up to 80 simultaneous queries, any text; switching queries is free

## Requirements

- iOS 16.0+, Xcode 15.0+
- Physical device (camera + Neural Engine)

## Quick Start

1. Open `YOLOEDemo.xcodeproj` in Xcode
2. Select your development team
3. Build and run on a physical device

## Re-converting Models (Optional)

```bash
pip install ultralytics coremltools clip-anytorch
python convert_models.py                # yoloe-11s-seg
python convert_models.py --size 11m-seg
```

This regenerates `yoloe_detector.mlpackage`, `reprta.mlpackage`, and
`clip_vocab.json`. The MobileCLIP text encoder is reused from Apple's export.

## Usage

1. Enter comma-separated object names (e.g. `person, dog, car`)
2. Tap the search button or press return
3. Switch Camera / Photo / Video with the bottom tabs
4. Adjust the confidence slider; in Photo mode masks re-render instantly
