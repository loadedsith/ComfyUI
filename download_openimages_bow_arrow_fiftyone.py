#!/usr/bin/env python3
"""
Alternative method: Download bow and arrow images using FiftyOne library
This is simpler but requires installing: pip install fiftyone
Copies images and captions to ComfyUI input folder in the correct format
"""

import sys
import shutil
from pathlib import Path

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("✗ FiftyOne not installed. Install with: pip install fiftyone")
    print("  Or use the manual download script: download_openimages_bow_arrow.py")
    sys.exit(1)

# Try to get ComfyUI input directory, fallback to local output
try:
    import folder_paths
    INPUT_DIR = Path(folder_paths.get_input_directory())
    OUTPUT_DIR = INPUT_DIR / "bow_arrow"
except:
    INPUT_DIR = None
    OUTPUT_DIR = Path("openimages_bow_arrow_fiftyone")


def main():
    print("=" * 70)
    print("Open Images Dataset - Bow and Arrow Downloader (FiftyOne)")
    print("=" * 70)

    print("\nDownloading first 10 images with 'Bow and arrow' label (class 59) from validation set...")
    print("Including bounding boxes...")
    print("Note: 'Bow and arrow' is class 59 in Open Images V7")
    print("Note: FiftyOne doesn't support 'localized_narratives' - captions will be empty")

    try:
        # Load dataset with specific class filter
        # "Bow and arrow" is class 59 in Open Images V7 (per Ultralytics docs)
        # Note: FiftyOne doesn't support localized_narratives label type
        # Supported types: ['classifications', 'detections', 'points', 'relationships', 'segmentations']
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            label_types=["detections"],  # bounding boxes only
            classes=["Bow and arrow"],  # Class 59 in Open Images V7
            max_samples=10,  # First 10 images
        )

        print(f"\n✓ Downloaded {len(dataset)} images")
        print(f"Dataset info:")
        print(f"  Name: {dataset.name}")

        # Create output directory (ComfyUI input folder if available)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if INPUT_DIR:
            print(f"\n📁 Copying to ComfyUI input folder: {OUTPUT_DIR}")
        else:
            print(f"\n📁 Saving to: {OUTPUT_DIR}")

        # Copy images and create matching caption files
        print(f"\nCopying images and creating caption files...")
        caption_count = 0
        image_count = 0

        for sample in dataset:
            source_image_path = Path(sample.filepath)
            image_filename = source_image_path.name

            # Copy image to output directory
            dest_image_path = OUTPUT_DIR / image_filename
            shutil.copy2(source_image_path, dest_image_path)
            image_count += 1

            # Create matching .txt caption file (same base name as image)
            caption_filename = OUTPUT_DIR / f"{source_image_path.stem}.txt"

            # Try to get caption from any available field
            caption_text = ""

            # Check various possible caption fields
            if hasattr(sample, 'captions') and sample.captions:
                captions = sample.captions
                if isinstance(captions, list) and len(captions) > 0:
                    caption_text = captions[0]
                else:
                    caption_text = str(captions)
            elif hasattr(sample, 'caption') and sample.caption:
                caption_text = sample.caption
            elif hasattr(sample, 'localized_narratives') and sample.localized_narratives:
                # Try anyway, even though it's not officially supported
                narratives = sample.localized_narratives
                if isinstance(narratives, list) and len(narratives) > 0:
                    first_narrative = narratives[0]
                    if hasattr(first_narrative, 'text'):
                        caption_text = first_narrative.text
                    elif isinstance(first_narrative, dict):
                        caption_text = first_narrative.get('text', '')

            # Save caption file (even if empty, to maintain file correspondence)
            with open(caption_filename, 'w', encoding='utf-8') as f:
                f.write(caption_text.strip() if caption_text else "")

            if caption_text:
                caption_count += 1

        print(f"✓ Copied {image_count} images")
        if caption_count > 0:
            print(f"✓ Created {caption_count} caption files with text")
        else:
            print(f"⚠ Created {image_count} caption files but they're empty (FiftyOne limitation)")
            print(f"  To get captions, use the manual download script instead.")

        print(f"\n{'=' * 70}")
        if INPUT_DIR:
            print(f"✓ Images and captions saved to ComfyUI input folder:")
            print(f"  {OUTPUT_DIR}")
            print(f"\n  You can now use 'Load Image Text Dataset from Folder' node")
            print(f"  with folder: 'bow_arrow'")
        else:
            print(f"✓ Images and captions saved to:")
            print(f"  {OUTPUT_DIR}")
        print(f"{'=' * 70}")

        # Optionally launch the FiftyOne app to view images
        print("\nLaunching FiftyOne app to view images...")
        session = fo.launch_app(dataset)
        print("Close the app window when done viewing.")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
