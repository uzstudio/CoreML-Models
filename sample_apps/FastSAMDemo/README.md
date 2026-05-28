# FastSAMDemo

**"Segment everything"** with [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) on Core ML —
real-time camera, photo tap-to-pick, and offline video.

FastSAM is a YOLOv8-seg instance segmenter (not a SAM encoder/prompt-decoder), so one forward
pass segments every object; a tap simply *selects* one. It's the fastest of the SAM-family
models in this zoo for interactive and real-time use (~30 fps on-device with FastSAM-s @512).

| Mode | What it does |
|------|--------------|
| **Camera** | Live instance-segmentation overlay at the camera frame rate (FPS shown), drawn on a `CALayer` over an `AVCaptureVideoPreviewLayer`. Toggle segmentation on/off. |
| **Photo**  | Pick an image → coloured "everything" map → tap any object to isolate its mask. |
| **Video**  | Pick a clip → every frame is segmented (frames oriented upright + colour-managed first) → masks burned into a new H.264 .mp4 → preview + Save to Photos. |

Shared controls (top): **Resolution** 320 / 512 / 640, and **Conf** / **Max objects** sliders.
A lightweight IoU **tracker** keeps each object's colour stable across frames (camera + video).

## Reusing the core

This demo does **not** reimplement FastSAM — it depends on the shared **FastSAM engine in
[SamKit](https://github.com/john-rocky/SamKit)** (`FastSamSession`, `SAMKit` product) via a
local Swift Package reference, the same engine the SamKit sample app and the CoreMLModelsApp
hub use. The project references the package at `../../../SamKit/runtime/apple`, so it expects
**`CoreML-Models/` and `SamKit/` to sit side by side** (both cloned into the same parent
folder, e.g. `~/Downloads/`):

```
~/Downloads/
├── CoreML-Models/sample_apps/FastSAMDemo/   ← this project
└── SamKit/runtime/apple/                     ← Swift package (SAMKit)
```

If your layout differs, open the project in Xcode → File ▸ Add Package Dependencies ▸ Add
Local… and point it at your `SamKit/runtime/apple` folder (or use the GitHub URL).

## Models

Per repo policy `.mlpackage` files are **not committed**. Generate the three resolutions and
drag them into the `FastSAMDemo/` group in Xcode (Target Membership: FastSAMDemo):

```bash
python ../../conversion_scripts/convert_fastsam.py --size s --input-size 320 512 640
# → FastSAM_s_320.mlpackage / FastSAM_s_512.mlpackage / FastSAM_s_640.mlpackage (~23 MB each)
```

These are **ImageType** (CVPixelBuffer) models with `scale=1/255` baked in. The Resolution
picker loads `FastSAM_s_<size>`. Until a model is added the app shows a "not bundled" notice
instead of crashing. (FastSAM-x is available via `--size x` if you want higher quality.)

## Run

```bash
open FastSAMDemo.xcodeproj
```

Build to a **physical device** (camera + Neural Engine; the simulator has no camera). Pick your
signing team in Signing & Capabilities. The camera path defaults to conf 0.5 / max 40 to keep
the frame rate up; lower **Conf** to surface more objects.

## Notes / gotchas (the hard-won ones)

- **FP16 models emit Float16 outputs** — `FastSamSession.readFloats` bulk-converts with
  `vImageConvert_Planar16FtoPlanarF`; reading element-by-element cost ~170 ms/frame.
- **Video frames must be oriented upright + sRGB colour-managed** before the model (the camera
  is upright already); feeding raw/sideways/un-managed frames wrecks detection.
- Mask assembly is one batched `sgemm` at proto resolution, composited and upscaled once.
