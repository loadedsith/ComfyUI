#!/usr/bin/env python3
"""
Download bow and arrow images from Open Images Dataset V7
Downloads the first 10 images with bounding box annotations and captions

Note: "Bow and arrow" is class 59 in Open Images V7 (per Ultralytics documentation)
"""

import csv
import urllib.request
import os
import random
from pathlib import Path
import subprocess
import sys

# Open Images dataset URLs
CLASS_DESCRIPTIONS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
# Image labels format (simpler, may be easier to access)
IMAGE_LABELS_TRAIN_URL = "https://storage.googleapis.com/openimages/v7/oidv7-train-images-with-labels-with-rotation.csv"
IMAGE_LABELS_VALIDATION_URL = "https://storage.googleapis.com/openimages/v7/oidv7-validation-images-with-labels-with-rotation.csv"
IMAGE_LABELS_TEST_URL = "https://storage.googleapis.com/openimages/v7/oidv7-test-images-with-labels-with-rotation.csv"
# Bounding box annotations (may require different access)
BOXES_TRAIN_URL = "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-bbox.csv"
BOXES_VALIDATION_URL = "https://storage.googleapis.com/openimages/v7/oidv7-validation-annotations-bbox.csv"
BOXES_TEST_URL = "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-bbox.csv"
# Localized narratives (captions)
NARRATIVES_TRAIN_URL = "https://storage.googleapis.com/openimages/v7/oidv7-train-localized-narratives.csv"
NARRATIVES_VALIDATION_URL = "https://storage.googleapis.com/openimages/v7/oidv7-validation-localized-narratives.csv"
NARRATIVES_TEST_URL = "https://storage.googleapis.com/openimages/v7/oidv7-test-localized-narratives.csv"

DOWNLOADER_SCRIPT_URL = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
OUTPUT_DIR = Path("openimages_bow_arrow")
IMAGE_LIST_FILE = OUTPUT_DIR / "image_list.txt"


def download_file(url, output_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    try:
        # Create a request with headers to avoid 403 errors
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


def find_class_mid(class_descriptions_file, search_terms):
    """
    Find the MID (Machine ID) for a class by searching class descriptions.
    search_terms: list of terms to search for (e.g., ['bow', 'arrow'])
    """
    print(f"\nSearching for class MID matching: {search_terms}")
    matches = []

    with open(class_descriptions_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                mid = row[0]
                display_name = row[1].lower()

                # Check if any search term matches
                if any(term.lower() in display_name for term in search_terms):
                    matches.append((mid, row[1]))
                    print(f"  Found: {row[1]} (MID: {mid})")

    if not matches:
        print("✗ No matching classes found!")
        return None

    # Prefer exact matches or most relevant
    # Look for "bow" and "arrow" together
    for mid, name in matches:
        name_lower = name.lower()
        if 'bow' in name_lower and 'arrow' in name_lower:
            print(f"\n✓ Selected: {name} (MID: {mid})")
            return mid

    # Otherwise return first match
    print(f"\n✓ Selected: {matches[0][1]} (MID: {matches[0][0]})")
    return matches[0][0]


def filter_images_by_class(annotations_file, class_mid, max_images=None, use_image_labels=False):
    """
    Filter image IDs that contain the specified class.
    Returns list of tuples: (split, image_id)

    use_image_labels: If True, expects image labels format (ImageID,Source,LabelName,Confidence)
                      If False, expects bounding box format (ImageID,LabelName,...)
    max_images: Maximum number of images to return. If None, returns all available.
    """
    print(f"\nFiltering images for class MID: {class_mid}")
    if max_images:
        print(f"  Limiting to {max_images} images")
    image_ids = []
    seen_ids = set()  # Avoid duplicates

    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_name = row.get('LabelName')
                if label_name == class_mid:
                    image_id = row.get('ImageID')
                    if image_id and image_id not in seen_ids:
                        seen_ids.add(image_id)
                        # Determine split from filename
                        if 'train' in str(annotations_file):
                            split = 'train'
                        elif 'validation' in str(annotations_file):
                            split = 'validation'
                        elif 'test' in str(annotations_file):
                            split = 'test'
                        else:
                            split = 'train'  # default

                        image_ids.append((split, image_id))

                        if max_images and len(image_ids) >= max_images:
                            break
    except Exception as e:
        print(f"✗ Error reading annotations: {e}")
        import traceback
        traceback.print_exc()
        return []

    print(f"✓ Found {len(image_ids)} images")
    return image_ids


def create_image_list_file(image_ids, output_file):
    """Create the image list file in format: split/image_id"""
    print(f"\nCreating image list file: {output_file}")
    with open(output_file, 'w') as f:
        for split, image_id in image_ids:
            f.write(f"{split}/{image_id}\n")
    print(f"✓ Created file with {len(image_ids)} image IDs")


def extract_captions(narratives_file, image_ids, output_dir):
    """
    Extract captions from localized narratives file for given image IDs.
    Returns dict mapping image_id -> caption_text
    """
    print(f"\nExtracting captions from: {narratives_file}")
    captions = {}

    if not narratives_file.exists():
        print(f"  ✗ Narratives file not found: {narratives_file}")
        return captions

    # Create a set of image IDs we're looking for
    target_ids = {img_id for _, img_id in image_ids}

    try:
        with open(narratives_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get('ImageID')
                if image_id in target_ids:
                    # Localized narratives format: ImageID,Text,...
                    caption_text = row.get('Text', '').strip()
                    if caption_text:
                        # Store caption (may have multiple narratives per image, keep first or concatenate)
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


def save_captions(captions, image_ids, images_dir, captions_dir):
    """
    Save captions as .txt files matching image filenames.
    Images are downloaded with format: split_imageid.jpg
    """
    print(f"\nSaving captions to: {captions_dir}")
    captions_dir.mkdir(exist_ok=True)

    saved_count = 0
    for split, image_id in image_ids:
        # Find the corresponding image file
        # Images are typically saved as: split_imageid.jpg or just imageid.jpg
        image_patterns = [
            f"{split}_{image_id}.jpg",
            f"{image_id}.jpg",
            f"{split}_{image_id}.png",
            f"{image_id}.png",
        ]

        image_file = None
        for pattern in image_patterns:
            potential_file = images_dir / pattern
            if potential_file.exists():
                image_file = potential_file
                break

        if not image_file:
            # Try to find any file starting with image_id
            for img_file in images_dir.glob(f"*{image_id}*"):
                image_file = img_file
                break

        if image_file:
            caption_text = captions.get(image_id, "")
            caption_file = captions_dir / f"{image_file.stem}.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption_text)
            saved_count += 1

    print(f"  ✓ Saved {saved_count} caption files")


def download_images(image_list_file, download_folder, num_processes=5):
    """Download images using the Open Images downloader script."""
    print(f"\nDownloading images to: {download_folder}")

    # Check if downloader.py exists
    downloader_script = Path("downloader.py")
    if not downloader_script.exists():
        print("Downloading downloader.py script...")
        if not download_file(DOWNLOADER_SCRIPT_URL, downloader_script):
            print("✗ Failed to download downloader.py")
            return False

    # Make downloader executable
    os.chmod(downloader_script, 0o755)

    # Run the downloader
    cmd = [
        sys.executable,
        str(downloader_script),
        str(image_list_file),
        "--download_folder", str(download_folder),
        "--num_processes", str(num_processes)
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running downloader: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def main():
    print("=" * 70)
    print("Open Images Dataset - Bow and Arrow Image Downloader")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Download class descriptions
    class_descriptions_file = OUTPUT_DIR / "class-descriptions.csv"
    if not class_descriptions_file.exists():
        if not download_file(CLASS_DESCRIPTIONS_URL, class_descriptions_file):
            print("✗ Failed to download class descriptions")
            return
    else:
        print(f"✓ Using existing class descriptions: {class_descriptions_file}")

    # Step 2: Find the MID for bow and arrow
    # Note: "Bow and arrow" is class 59 in Open Images V7 (per Ultralytics docs)
    search_terms = ['bow', 'arrow']
    class_mid = find_class_mid(class_descriptions_file, search_terms)

    if not class_mid:
        print("\n✗ Could not find bow and arrow class. Available classes:")
        # Show some examples
        with open(class_descriptions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 20:
                    print(f"  {row[0]}: {row[1]}")
        return

    # Step 3: Try downloading image labels first (simpler format, may be easier to access)
    # Image labels format: ImageID,Source,LabelName,Confidence
    print("\nTrying image labels format first (simpler)...")
    annotations_file = OUTPUT_DIR / "validation-image-labels.csv"
    if not annotations_file.exists():
        print(f"\nDownloading validation image labels...")
        if not download_file(IMAGE_LABELS_VALIDATION_URL, annotations_file):
            print("✗ Failed to download validation image labels, trying bounding boxes...")
            annotations_file = OUTPUT_DIR / "validation-annotations-bbox.csv"
            if not annotations_file.exists():
                if not download_file(BOXES_VALIDATION_URL, annotations_file):
                    print("✗ Failed to download validation annotations")
                    return
    else:
        print(f"✓ Using existing annotations: {annotations_file}")

    # Step 4: Filter images (get first 10)
    image_ids = filter_images_by_class(annotations_file, class_mid, max_images=10, use_image_labels=True)

    if not image_ids:
        print("\n✗ No images found with bow and arrow annotations in validation set")
        print("  Trying train set (larger but takes longer to download)...")
        train_annotations_file = OUTPUT_DIR / "train-image-labels.csv"
        if not train_annotations_file.exists():
            print("  This will download a large file. Auto-continuing... (Ctrl+C to cancel)")
            try:
                if not download_file(IMAGE_LABELS_TRAIN_URL, train_annotations_file):
                    # Fallback to bounding boxes
                    train_annotations_file = OUTPUT_DIR / "train-annotations-bbox.csv"
                    if not train_annotations_file.exists():
                        if not download_file(BOXES_TRAIN_URL, train_annotations_file):
                            return
            except KeyboardInterrupt:
                print("\n✗ Cancelled by user")
                return
        image_ids = filter_images_by_class(train_annotations_file, class_mid, max_images=10, use_image_labels=True)

    if not image_ids:
        print("✗ Still no images found")
        return

    if len(image_ids) < 10:
        print(f"  Found {len(image_ids)} images (requested 10)")
    else:
        print(f"  Found {len(image_ids)} images (limited to first 10)")

    # Step 5: Create image list file
    create_image_list_file(image_ids, IMAGE_LIST_FILE)

    # Step 6: Download localized narratives (captions)
    print("\n" + "=" * 70)
    print("Downloading captions (localized narratives)...")
    narratives_file = None
    for split, _ in image_ids[:1]:  # Check first image to determine split
        if split == 'validation':
            narratives_file = OUTPUT_DIR / "validation-localized-narratives.csv"
            narratives_url = NARRATIVES_VALIDATION_URL
        elif split == 'train':
            narratives_file = OUTPUT_DIR / "train-localized-narratives.csv"
            narratives_url = NARRATIVES_TRAIN_URL
        elif split == 'test':
            narratives_file = OUTPUT_DIR / "test-localized-narratives.csv"
            narratives_url = NARRATIVES_TEST_URL
        break

    if narratives_file and not narratives_file.exists():
        print(f"Downloading narratives from: {narratives_url}")
        if not download_file(narratives_url, narratives_file):
            print("⚠ Warning: Failed to download narratives. Continuing without captions...")
            narratives_file = None
    elif narratives_file:
        print(f"✓ Using existing narratives file: {narratives_file}")

    # Step 7: Download images
    download_folder = OUTPUT_DIR / "images"
    download_folder.mkdir(exist_ok=True)

    if download_images(IMAGE_LIST_FILE, download_folder):
        print(f"\n✓ Successfully downloaded images to: {download_folder}")

        # Step 8: Extract and save captions
        if narratives_file:
            captions = extract_captions(narratives_file, image_ids, OUTPUT_DIR)
            if captions:
                captions_dir = OUTPUT_DIR / "captions"
                save_captions(captions, image_ids, download_folder, captions_dir)

        print(f"\n{'=' * 70}")
        print(f"✓ Download complete!")
        print(f"  Images: {download_folder}")
        if narratives_file:
            print(f"  Captions: {OUTPUT_DIR / 'captions'}")
        print(f"{'=' * 70}")
    else:
        print("\n✗ Failed to download images")


if __name__ == "__main__":
    main()

