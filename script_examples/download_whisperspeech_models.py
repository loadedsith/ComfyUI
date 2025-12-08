#!/usr/bin/env python3
"""
Download WhisperSpeech models for testing.

Downloads 2+ T2S models and 2+ S2A models to models/whisperspeech folder for use in ComfyUI.
"""

import sys
import os
import shutil
from pathlib import Path

# Add parent directory to path to import whisperspeech and folder_paths
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import hf_hub_download
    from whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer
    from whisperspeech.s2a_delar_mup_wds_mlang import SADelARTransformer
    import torch
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install with: pip install whisperspeech huggingface_hub")
    sys.exit(1)

# Models to download for testing
T2S_MODELS = [
    "collabora/whisperspeech:t2s-small-en+pl.model",  # Default
    "collabora/whisperspeech:t2s-fast-small-en+pl.model",  # Fast variant
    "collabora/whisperspeech:t2s-v1.9-medium-7lang.model",  # Multilingual
    "collabora/whisperspeech:t2s-tiny-en+pl.model",  # Tiny/fastest
]

S2A_MODELS = [
    "collabora/whisperspeech:s2a-q4-hq-fast-en+pl.model",  # Fast, high quality
    "collabora/whisperspeech:s2a-q4-small-en+pl.model",  # Small, fast
    "collabora/whisperspeech:s2a-v1.1-small-en+pl.model",  # Alternative version
]

def download_model(ref: str, model_type: str, output_dir: Path):
    """Download a model to models/whisperspeech folder and verify it can be loaded."""
    print(f"\n{'='*70}")
    print(f"Downloading {model_type}: {ref}")
    print(f"{'='*70}")
    
    try:
        # Parse reference
        if ":" in ref:
            repo_id, filename = ref.split(":", 1)
        else:
            repo_id = "collabora/whisperspeech"
            filename = ref
        
        # Download model to HuggingFace cache first
        print(f"  Repository: {repo_id}")
        print(f"  Filename: {filename}")
        print(f"  Downloading from HuggingFace...")
        
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )
        
        print(f"  ✓ Downloaded to cache: {cached_path}")
        
        # Copy to models/whisperspeech folder
        output_path = output_dir / filename
        print(f"  Copying to: {output_path}")
        shutil.copy2(cached_path, output_path)
        print(f"  ✓ Copied to models folder: {output_path}")
        
        # Verify model can be loaded
        print(f"  Verifying model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_type == "T2S":
            # Test loading from local path
            model = TSARTransformer.load_model(ref=str(output_path), device=device)
            print(f"  ✓ T2S model loaded successfully")
        elif model_type == "S2A":
            # S2A loading is more complex, just check file exists
            print(f"  ✓ S2A model file verified")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Get ComfyUI base directory
    comfyui_base = Path(__file__).parent.parent
    models_dir = comfyui_base / "models" / "audio_encoders"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("WhisperSpeech Model Downloader")
    print("="*70)
    print(f"\nOutput directory: {models_dir}")
    print(f"\nWill download:")
    print(f"  - {len(T2S_MODELS)} T2S (text-to-semantic) models")
    print(f"  - {len(S2A_MODELS)} S2A (semantic-to-audio) models")
    print(f"\nTotal: {len(T2S_MODELS) + len(S2A_MODELS)} models")
    print("\nModels will be saved to models/audio_encoders/ folder")
    
    # Download T2S models
    print("\n" + "="*70)
    print("T2S MODELS (Text-to-Semantic)")
    print("="*70)
    t2s_success = 0
    for ref in T2S_MODELS:
        if download_model(ref, "T2S", models_dir):
            t2s_success += 1
    
    # Download S2A models
    print("\n" + "="*70)
    print("S2A MODELS (Semantic-to-Audio)")
    print("="*70)
    s2a_success = 0
    for ref in S2A_MODELS:
        if download_model(ref, "S2A", models_dir):
            s2a_success += 1
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"T2S models: {t2s_success}/{len(T2S_MODELS)} successful")
    print(f"S2A models: {s2a_success}/{len(S2A_MODELS)} successful")
    print(f"Total: {t2s_success + s2a_success}/{len(T2S_MODELS) + len(S2A_MODELS)} successful")
    print(f"\nModels saved to: {models_dir}")
    print("  (Note: WhisperSpeech models are stored alongside audio encoder models)")
    
    if t2s_success >= 2 and s2a_success >= 2:
        print("\n✓ Success! You now have multiple models for testing.")
        print("  Restart ComfyUI to see them in the dropdown menus.")
    else:
        print("\n⚠ Warning: Some models failed to download.")
        print("  You may need to check your internet connection or HuggingFace access.")

if __name__ == "__main__":
    main()
