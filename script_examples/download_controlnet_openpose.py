#!/usr/bin/env python3
"""
Download ControlNet OpenPose model for ComfyUI.

Downloads a ControlNet model trained on OpenPose data that can work with DWPose outputs.
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
    print("Error: Missing required package: huggingface-hub")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)

# ControlNet OpenPose models - SD 1.5 compatible
CONTROLNET_MODELS = [
    {
        "repo_id": "lllyasviel/control_v11p_sd15_openpose",
        "filename": "diffusion_pytorch_model.safetensors",
        "model_name": "control_v11p_sd15_openpose.safetensors",
        "model_type": "SD 1.5 OpenPose ControlNet"
    },
    {
        "repo_id": "lllyasviel/control_v11p_sd15_openpose",
        "filename": "diffusion_pytorch_model.safetensors",
        "model_name": "control_v11p_sd15_openpose.safetensors",
        "model_type": "SD 1.5 OpenPose ControlNet (alternative)"
    }
]

def download_model(repo_id: str, filename: str, model_name: str, model_type: str, output_dir: Path):
    """Download a ControlNet model file."""
    print(f"\n{'='*70}")
    print(f"Downloading {model_type}")
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

        # Copy to models/controlnet folder with the model_name
        output_path = output_dir / model_name
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

    # ControlNet models go in models/controlnet
    output_dir = comfyui_base / "models" / "controlnet"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("ControlNet OpenPose Model Downloader")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nWill download:")
    print(f"  - SD 1.5 OpenPose ControlNet model")
    print(f"\nThis model works with DWPose/OpenPose pose data.")

    # Download the first model (they're duplicates, just download one)
    model_info = CONTROLNET_MODELS[0]
    success = download_model(
        model_info["repo_id"],
        model_info["filename"],
        model_info["model_name"],
        model_info["model_type"],
        output_dir
    )

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    if success:
        print(f"✓ Success! ControlNet OpenPose model downloaded.")
        print(f"\nModels saved to: {output_dir}")
        print("\nTo use:")
        print("  1. Install ControlNet Auxiliary Preprocessors custom node (if not already installed)")
        print("  2. Use DWPose node to extract pose from images")
        print("  3. Load this ControlNet model in 'Load ControlNet Model' node")
        print("  4. Connect pose image to ControlNet Apply node")
    else:
        print("⚠ Warning: Model download failed.")
        print("  You may need to check your internet connection or HuggingFace access.")

if __name__ == "__main__":
    main()

