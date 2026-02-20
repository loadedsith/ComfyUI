#!/usr/bin/env python3
"""
Count how many "Bow and arrow" (class 59) images are in Open Images V7
"""

import sys
from pathlib import Path

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("✗ FiftyOne not installed. Install with: pip install fiftyone")
    sys.exit(1)

def main():
    print("=" * 70)
    print("Counting 'Bow and arrow' (class 59) images in Open Images V7")
    print("=" * 70)

    counts = {}

    for split in ["train", "validation", "test"]:
        print(f"\nChecking {split} split...")
        try:
            # Load dataset without limiting samples to count all
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=["Bow and arrow"],  # Class 59
            )

            count = len(dataset)
            counts[split] = count
            print(f"  ✓ {split}: {count:,} images")

        except Exception as e:
            print(f"  ✗ Error loading {split}: {e}")
            counts[split] = 0

    total = sum(counts.values())

    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"  Train:      {counts.get('train', 0):,} images")
    print(f"  Validation: {counts.get('validation', 0):,} images")
    print(f"  Test:       {counts.get('test', 0):,} images")
    print(f"  {'-' * 68}")
    print(f"  TOTAL:     {total:,} images")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()


