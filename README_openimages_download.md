# Downloading Bow and Arrow Images from Open Images Dataset

Two methods are available to download bow and arrow images with captions from Open Images Dataset V7.

**Note:** "Bow and arrow" is **class 59** in Open Images V7 (per [Ultralytics documentation](https://docs.ultralytics.com/datasets/detect/open-images-v7/)).

## Method 1: Using FiftyOne (Recommended - Simplest)

The FiftyOne library handles authentication and downloads automatically.

### Installation
```bash
pip install fiftyone
```

### Usage
```bash
python3 download_openimages_bow_arrow_fiftyone.py
```

This will:
- Download the first 10 images with "Bow and arrow" labels (class 59) from the validation set
- Include bounding box annotations
- Include captions (localized narratives)
- Export images to `openimages_bow_arrow_fiftyone/images/`
- Save captions as `.txt` files to `openimages_bow_arrow_fiftyone/captions/`
- Launch the FiftyOne app for visualization

## Method 2: Manual Download Script

The manual script (`download_openimages_bow_arrow.py`) attempts to:
1. Download class descriptions to find the "Bow and arrow" MID (`/m/01g3x7`)
2. Download annotation files
3. Filter for images with that class
4. Use the Open Images downloader script to fetch images

**Note**: The direct Google Storage URLs may require authentication (403 errors). The script will attempt downloads but may fail. If it fails, use Method 1 (FiftyOne) instead.

The script will also:
- Download localized narratives (captions) for the images
- Save captions as `.txt` files matching image filenames

### Usage
```bash
python3 download_openimages_bow_arrow.py
```

## Class Information

- **Class Name**: Bow and arrow
- **MID (Machine ID)**: `/m/01g3x7`
- **Found in**: Validation and Train sets

## Output

Images and captions will be saved to:
- Method 1:
  - Images: `openimages_bow_arrow_fiftyone/images/`
  - Captions: `openimages_bow_arrow_fiftyone/captions/` (`.txt` files matching image names)
- Method 2:
  - Images: `openimages_bow_arrow/images/`
  - Captions: `openimages_bow_arrow/captions/` (`.txt` files matching image names)

## Troubleshooting

If you get 403 errors with the manual script:
- Use the FiftyOne method instead (Method 1)
- Or download annotation files manually from the [Open Images website](https://storage.googleapis.com/openimages/web/download_v7.html) and place them in the `openimages_bow_arrow/` directory

