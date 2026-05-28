# Adding FastSAM to the hub (CoreMLModelsApp)

The hub is **data-driven by the remote manifest** (`models.json` on HuggingFace, see
`Paths.manifestURLString`), so FastSAM appears once two things are true: (1) the app has the
demo template compiled in, and (2) the published manifest has a FastSAM entry pointing at the
uploaded `.mlpackage`. The code for (1) is already scaffolded in this repo; the steps below
finish the wiring. These are kept out of the `.xcodeproj` deliberately so the shipping app
keeps building until you opt in.

## 1. Add the two scaffolded files to the target

Both files are already on disk; just add them to the **CoreMLModelsApp** target (drag into the
matching Xcode group, or check "Target Membership"):

- `CoreMLModelsApp/SAMKit/FastSAM.swift` — the FastSAM engine, vendored into the app module
  (same source as SamKit's `FastSamSession`, with `public` dropped to match the other vendored
  SAMKit files). This is the "reuse the core" copy for the hub.
- `CoreMLModelsApp/Templates/FastSamDemoView.swift` — the "segment everything" + tap demo,
  modelled on `SegmentAnythingDemoView`.

## 2. Register the template in `DemoLauncherView.swift`

Add one case to the `switch model.demo.template` block:

```swift
case "fast_sam":
    FastSamDemoView(model: model)
```

## 3. Convert and upload the model

```bash
python conversion_scripts/convert_fastsam.py --size s    # FastSAM_s.mlpackage (~23 MB)
```

Zip it (`FastSAM_s.mlpackage.zip`) and upload to the `mlboydaisuke/coreml-zoo` HF repo under a
`fastsam/` path, the same way the other models are hosted.

## 4. Add the manifest entry

Add this to `models.json` (use the same `add_*_to_live_manifest.py` flow as the other models so
`size_bytes` / `sha256` are filled from the uploaded file). `FastSAM_x` can be added as a second
`optional` file or a separate entry.

```json
{
  "id": "fastsam_s",
  "name": "FastSAM-s",
  "subtitle": "CASIA-IVA-Lab, 2023",
  "category_id": "segmentation",
  "description_md": "Fast Segment Anything (YOLOv8-seg). One forward pass segments every object; tap to pick one. The fastest SAM-family model for 'segment everything' and real-time use.",
  "demo": { "template": "fast_sam", "config": { "input_size": 640, "output_type": "mask" } },
  "files": [
    {
      "name": "FastSAM_s.mlpackage.zip",
      "url": "https://huggingface.co/mlboydaisuke/coreml-zoo/resolve/main/fastsam/FastSAM_s.mlpackage.zip",
      "archive": "zip",
      "size_bytes": 0,
      "sha256": "<fill from upload>",
      "compute_units": "cpuAndNeuralEngine",
      "kind": "model"
    }
  ],
  "requirements": { "min_ios": "16.0", "min_ram_mb": 300 },
  "license": { "name": "AGPL-3.0", "url": "https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/LICENSE" },
  "upstream": { "name": "CASIA-IVA-Lab/FastSAM", "url": "https://github.com/CASIA-IVA-Lab/FastSAM", "year": 2023 },
  "conversion_script_url": "https://github.com/john-rocky/CoreML-Models/blob/master/conversion_scripts/convert_fastsam.py"
}
```

> Note the **AGPL-3.0** license (Ultralytics YOLOv8 lineage) — unlike the Apache-2.0
> MobileSAM / SAM2 entries.
