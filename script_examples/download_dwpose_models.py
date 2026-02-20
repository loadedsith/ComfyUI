#!/usr/bin/env python3
"""
Download DWPose models for ComfyUI.

Downloads the required ONNX models:
- dw-ll_ucoco_384.onnx (pose estimation model)
- yolox_l.onnx (detection model)
"""

import sys
import os
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: Missing required package: huggingface_hub")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)

# DWPose models - downloaded from yzd-v/DWPose repository
DWPOSE_MODELS = [
    {
        "repo_id": "yzd-v/DWPose",
        "filename": "dw-ll_ucoco_384.onnx",
        "model_type": "Pose Model"
    },
    {
        "repo_id": "yzd-v/DWPose",
        "filename": "yolox_l.onnx",
        "model_type": "Detection Model"
    }
]

def download_model(repo_id: str, filename: str, model_type: str, output_dir: Path):
    """Download a DWPose model file."""
    print(f"\n{'='*70}")
    print(f"Downloading {model_type}: {filename}")
    print(f"{'='*70}")

    try:
        print(f"  Repository: {repo_id}")
        print(f"  Filename: {filename}")
        print(f"  Downloading from HuggingFace...")

        # Download model to HuggingFace cache first
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )

        print(f"  ✓ Downloaded to cache: {cached_path}")

        # Copy to models/dwpose folder
        output_path = output_dir / filename
        print(f"  Copying to: {output_path}")
        shutil.copy2(cached_path, output_path)
        print(f"  ✓ Copied to models folder: {output_path}")

        # Verify file exists and has content
        if output_path.exists() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Model file verified ({size_mb:.2f} MB)")
            return True
        else:
            print(f"  ✗ Error: File verification failed")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Get ComfyUI base directory
    comfyui_base = Path(__file__).parent.parent

    # DWPose models are typically stored in models/annotators or models/dwpose
    # Check if annotators folder exists (common for ControlNet-related models)
    annotators_dir = comfyui_base / "models" / "annotators"
    dwpose_dir = comfyui_base / "models" / "dwpose"

    # Use annotators if it exists, otherwise create dwpose folder
    if annotators_dir.exists():
        output_dir = annotators_dir
        print(f"Using existing annotators directory: {output_dir}")
    else:
        output_dir = dwpose_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating dwpose directory: {output_dir}")

    print("="*70)
    print("DWPose Model Downloader")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nWill download:")
    for model_info in DWPOSE_MODELS:
        print(f"  - {model_info['model_type']}: {model_info['filename']}")
    print(f"\nTotal: {len(DWPOSE_MODELS)} models")

    # Download models
    success_count = 0
    for model_info in DWPOSE_MODELS:
        if download_model(
            model_info["repo_id"],
            model_info["filename"],
            model_info["model_type"],
            output_dir
        ):
            success_count += 1

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Models downloaded: {success_count}/{len(DWPOSE_MODELS)}")
    print(f"\nModels saved to: {output_dir}")

    if success_count == len(DWPOSE_MODELS):
        print("\n✓ Success! DWPose models are ready to use.")
        print("  If you're using a DWPose custom node, restart ComfyUI to see them.")
    else:
        print("\n⚠ Warning: Some models failed to download.")
        print("  You may need to check your internet connection or HuggingFace access.")
        print("  Alternative: Install the dwpose package: pip install dwpose")
        print("  Then models will be downloaded automatically on first use.")

if __name__ == "__main__":
    main()

