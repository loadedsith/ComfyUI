#!/usr/bin/env python3
"""
Generate caption files for bow and arrow images by downloading localized narratives
"""

import csv
import urllib.request
from pathlib import Path

# Open Images dataset URLs
NARRATIVES_VALIDATION_URL = "https://storage.googleapis.com/openimages/v7/oidv7-validation-localized-narratives.csv"

INPUT_DIR = Path("input/bow_arrow")


def download_file(url, output_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')

        with urllib.request.urlopen(req) as response:
            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False


def get_image_ids_from_folder(folder):
    """Extract image IDs from image filenames in the folder."""
    image_ids = []
    for img_file in folder.glob("*.jpg"):
        # Image ID is the filename without extension
        image_id = img_file.stem
        image_ids.append(image_id)
    return image_ids


def extract_captions(narratives_file, image_ids):
    """
    Extract captions from localized narratives file for given image IDs.
    Returns dict mapping image_id -> caption_text
    """
    print(f"\nExtracting captions from: {narratives_file}")
    captions = {}

    if not narratives_file.exists():
        print(f"  ✗ Narratives file not found: {narratives_file}")
        return captions

    target_ids = set(image_ids)

    try:
        with open(narratives_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get('ImageID')
                if image_id in target_ids:
                    # Localized narratives format: ImageID,Text,...
                    caption_text = row.get('Text', '').strip()
                    if caption_text:
                        # Store caption (may have multiple narratives per image, concatenate)
                        if image_id not in captions:
                            captions[image_id] = caption_text
                        else:
                            # If multiple narratives, append
                            captions[image_id] += " " + caption_text
    except Exception as e:
        print(f"  ✗ Error reading narratives: {e}")
        import traceback
        traceback.print_exc()

    print(f"  ✓ Extracted {len(captions)} captions")
    return captions


def write_caption_files(folder, captions):
    """Write caption text to .txt files matching image filenames."""
    print(f"\nWriting caption files to: {folder}")
    written = 0

    for img_file in folder.glob("*.jpg"):
        image_id = img_file.stem
        caption_file = folder / f"{image_id}.txt"

        caption_text = captions.get(image_id, "")
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption_text)

        if caption_text:
            written += 1
            print(f"  ✓ {image_id}.txt: {caption_text[:60]}...")
        else:
            print(f"  ⚠ {image_id}.txt: (no caption found)")

    print(f"\n✓ Written {written} caption files with text")


def main():
    print("=" * 70)
    print("Generate Captions for Bow and Arrow Images")
    print("=" * 70)

    if not INPUT_DIR.exists():
        print(f"✗ Input directory not found: {INPUT_DIR}")
        return

    # Get image IDs from existing images
    image_ids = get_image_ids_from_folder(INPUT_DIR)
    print(f"\nFound {len(image_ids)} images in {INPUT_DIR}")

    if not image_ids:
        print("✗ No images found!")
        return

    # Download localized narratives CSV
    narratives_file = Path("openimages_bow_arrow/validation-localized-narratives.csv")
    narratives_file.parent.mkdir(exist_ok=True)

    if not narratives_file.exists():
        print(f"\nDownloading localized narratives...")
        if not download_file(NARRATIVES_VALIDATION_URL, narratives_file):
            print("✗ Failed to download narratives")
            return
    else:
        print(f"\n✓ Using existing narratives file: {narratives_file}")

    # Extract captions
    captions = extract_captions(narratives_file, image_ids)

    if not captions:
        print("\n⚠ No captions found for these images")
        print("  This could mean:")
        print("  - These images don't have localized narratives")
        print("  - The narratives file format is different")
        return

    # Write caption files
    write_caption_files(INPUT_DIR, captions)

    print(f"\n{'=' * 70}")
    print("✓ Done! Caption files updated.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()


