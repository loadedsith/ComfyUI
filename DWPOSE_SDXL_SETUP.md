# DWPose + SDXL OpenPose ControlNet Setup Guide

## ✅ What's Installed

1. **Custom Node**: `comfyui_controlnet_aux` (ControlNet Auxiliary Preprocessors)
2. **DWPose Models**: `models/dwpose/` (dw-ll_ucoco_384.onnx, yolox_l.onnx)
3. **SDXL ControlNet Model**: `models/controlnet/controlnet-openpose-sdxl-1.0.safetensors` (4.7 GB)

## 🔍 Finding the Nodes in ComfyUI

After restarting ComfyUI, look for these nodes:

### DWPose Preprocessor Node
**Location**: Right-click → Add Node → **ControlNet Preprocessors** → **DWPose_Preprocessor**

**Alternative names you might see**:
- `DWPose Estimator`
- `DWPreprocessor`
- Or in an "AIO Aux Preprocessor" dropdown (select DWPose)

### ControlNet Nodes (Built-in)
- **Load ControlNet Model**: Right-click → Add Node → **loaders** → **Load ControlNet Model**
- **Apply ControlNet**: Right-click → Add Node → **conditioning** → **ControlNet ApplyAdvanced** (or **ControlNet Apply**)

## 📋 Minimal SDXL Workflow for Archery Poses

```
1. CheckpointLoaderSimple
   └─ MODEL, CLIP, VAE

2. Load Image (your pose reference photo)
   └─ IMAGE

3. DWPose_Preprocessor
   ├─ image: (from Load Image)
   ├─ detect_hand: "enable" (important for archery!)
   ├─ detect_body: "enable"
   ├─ detect_face: "enable" (optional)
   └─ resolution: 1024 (match your generation size)
   └─ OUTPUT: IMAGE (pose hint/stick figure)

4. Load ControlNet Model
   └─ control_net_name: "controlnet-openpose-sdxl-1.0.safetensors"
   └─ OUTPUT: CONTROL_NET

5. CLIPTextEncode (positive)
   ├─ clip: (from CheckpointLoaderSimple)
   └─ text: "person with bow and arrow, archery pose, ..."

6. ControlNet ApplyAdvanced
   ├─ conditioning: (from CLIPTextEncode positive)
   ├─ control_net: (from Load ControlNet Model)
   ├─ image: (from DWPose_Preprocessor)
   ├─ strength: 0.8-1.0 (start with 0.9)
   ├─ start_percent: 0.0
   └─ end_percent: 0.8-1.0 (start with 0.9)
   └─ OUTPUT: CONDITIONING (modified)

7. CLIPTextEncode (negative)
   ├─ clip: (from CheckpointLoaderSimple)
   └─ text: "blurry, distorted, bad anatomy"

8. EmptyLatentImage
   ├─ width: 1024
   ├─ height: 1024
   └─ batch_size: 1

9. KSampler
   ├─ model: (from CheckpointLoaderSimple)
   ├─ positive: (from ControlNet ApplyAdvanced)
   ├─ negative: (from CLIPTextEncode negative)
   ├─ latent_image: (from EmptyLatentImage)
   ├─ steps: 20-30 (start with 20)
   ├─ cfg: 7-8
   └─ sampler: "euler" or "dpmpp_2m"

10. VAEDecode
    ├─ samples: (from KSampler)
    └─ vae: (from CheckpointLoaderSimple)
```

## ⚙️ Archery-Specific Settings

### DWPose Preprocessor Settings
- **detect_hand**: `"enable"` ← CRITICAL for archery (bow hand, string hand)
- **detect_body**: `"enable"`
- **detect_face**: `"enable"` (optional, helps with head angle)
- **resolution**: Match your generation size (1024 for SDXL)

### ControlNet Apply Settings
- **strength**: `0.7-1.0` (start with `0.9`)
  - Higher = stricter pose adherence
  - Lower = more creative freedom
- **start_percent**: `0.0` (always start from beginning)
- **end_percent**: `0.7-1.0` (start with `0.9`)
  - **Warning**: If end_percent is too low (<0.6), pose collapses mid-generation
  - For archery, keep it high (0.8-1.0) to maintain pose integrity

### Generation Settings
- **Resolution**: Start with `1024x1024` (SDXL native)
- **Steps**: `20-30` (20 is often enough for pose tests)
- **CFG Scale**: `7-8` (SDXL works well at lower CFG)

## 🎯 Reference Image Tips for Archery

OpenPose/DWPose works best with:
- ✅ Clear silhouette (side view is gold)
- ✅ Minimal background clutter
- ✅ Not too small (don't feed 200px images)
- ✅ Full draw pose (bow arm extended, string hand at anchor)
- ❌ Avoid: extreme angles, heavy occlusion, tiny figures

## 🐛 Common Failure Modes (and fixes)

1. **Bow arm and string hand swap**
   - Fix: Better reference image, increase ControlNet strength

2. **Anchor drifts**
   - Fix: Enable hand detection, increase end_percent

3. **Elbow gets "invented"**
   - Fix: Better reference image, adjust strength

4. **Bow becomes banana-shaped**
   - Fix: Add second ControlNet (Canny/Lineart) for bow silhouette

## 💾 VRAM Tips (8GB GPU)

- Start with `1024x1024` resolution
- Use `20 steps` initially
- Only one ControlNet at first
- If OOM: reduce resolution to `768x768` or enable low-VRAM modes

## 🔄 Next Steps

1. **Restart ComfyUI** to load the custom node
2. Build the workflow above
3. Test with a clear archery reference image
4. Adjust strength/end_percent based on results
5. (Optional) Add second ControlNet for bow shape later

## 📝 Node Name Reference

If you can't find nodes, search for:
- **DWPose**: `DWPose_Preprocessor`, `DWPose Estimator`, `DWPreprocessor`
- **ControlNet Load**: `Load ControlNet Model`, `ControlNetLoader`
- **ControlNet Apply**: `ControlNet ApplyAdvanced`, `ControlNet Apply`, `Apply ControlNet`

