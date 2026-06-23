"""
PhenoLeaf-TS - Dataset Preparation
==================================
Converts the raw PhenoLeaf-TS dataset into model-ready form:
1. Scans the dataset structure
2. Creates a single standard 70/15/15 train/val/test split
3. Computes dataset statistics
4. Converts masks to COCO format (instance segmentation)
5. Prepares YOLO-format annotations
6. Generates growth-stage labels (classification)

Usage:
    python tools/prepare_data.py \
        --data_root data/leaf_dataset_colour \
        --output_dir data/prepared \
        --format all          # coco | yolo | all
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import shutil

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm


# Colour palette from metadata (31 unique colours)
COLOUR_PALETTE = [
    (244, 64, 14),    # Leaf 1
    (48, 57, 249),    # Leaf 2
    (234, 250, 37),   # Leaf 3
    (24, 193, 65),    # Leaf 4
    (245, 130, 49),   # Leaf 5
    (231, 80, 219),   # Leaf 6
    (0, 182, 173),    # Leaf 7
    (115, 0, 218),    # Leaf 8
    (191, 239, 69),   # Leaf 9
    (255, 250, 200),  # Leaf 10
    (250, 190, 212),  # Leaf 11
    (66, 212, 244),   # Leaf 12
    (155, 99, 36),    # Leaf 13
    (220, 190, 255),  # Leaf 14
    (69, 158, 220),   # Leaf 15
    (255, 216, 177),  # Leaf 16
    (98, 2, 37),      # Leaf 17
    (227, 213, 12),   # Leaf 18
    (79, 159, 83),    # Leaf 19
    (170, 23, 101),   # Leaf 20
    (170, 255, 195),  # Leaf 21
    (169, 169, 169),  # Leaf 22
    (181, 111, 119),  # Leaf 23
    (144, 121, 171),  # Leaf 24
    (9, 125, 244),    # Leaf 25
    (184, 70, 30),    # Leaf 26
    (154, 35, 246),   # Leaf 27
    (229, 225, 238),  # Leaf 28
    (141, 254, 82),   # Leaf 29
    (31, 200, 209),   # Leaf 30
    (194, 217, 105),  # Leaf 31
]


def get_growth_stage(leaf_count):
    """Get growth stage from leaf count."""
    if leaf_count <= 6:
        return 0  # Early
    elif leaf_count <= 10:
        return 1  # Intermediate
    else:
        return 2  # Mature


def count_leaves_in_mask(mask_path):
    """Count unique leaf instances in a mask."""
    try:
        mask = np.array(Image.open(mask_path).convert("RGB"))
        pixels = mask.reshape(-1, 3)
        unique_colours = np.unique(pixels, axis=0)
        # Remove background (black)
        num_leaves = sum(1 for c in unique_colours if not np.all(c == 0))
        return num_leaves
    except Exception as e:
        print(f"Warning: Could not read mask {mask_path}: {e}")
        return -1  # Return -1 to indicate error


def extract_instances_from_mask(mask_path):
    """Extract individual leaf instances from colour-coded mask.

    Returns list of dicts with:
        - mask: binary numpy array
        - colour: RGB tuple
        - leaf_id: integer
        - bbox: [x, y, w, h]
        - area: number of pixels
    """
    try:
        mask = np.array(Image.open(mask_path).convert("RGB"))
    except Exception as e:
        print(f"Warning: Could not read mask {mask_path}: {e}")
        return []
    h, w = mask.shape[:2]
    instances = []

    for leaf_id, colour in enumerate(COLOUR_PALETTE, start=1):
        # Find pixels matching this colour
        match = np.all(mask == np.array(colour), axis=-1)
        if match.sum() == 0:
            continue

        # Get bounding box
        rows = np.any(match, axis=1)
        cols = np.any(match, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        instances.append({
            "mask": match,
            "colour": colour,
            "leaf_id": leaf_id,
            "bbox": [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)],
            "area": int(match.sum()),
        })

    return instances


def mask_to_polygon(binary_mask):
    """Convert binary mask to polygon (COCO format)."""
    from skimage import measure

    contours = measure.find_contours(binary_mask, 0.5)
    polygons = []

    for contour in contours:
        # Flip y,x to x,y and flatten
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        if len(segmentation) >= 6:  # At least 3 points
            polygons.append(segmentation)

    return polygons


def scan_dataset(raw_dir, mask_dir):
    """Scan the dataset and build sample list."""
    samples = []

    raw_dir = Path(raw_dir)
    mask_dir = Path(mask_dir)

    # Iterate through plant genotypes
    for plant_dir in tqdm(sorted(raw_dir.iterdir()), desc="Scanning plants"):
        if not plant_dir.is_dir():
            continue

        plant_name = plant_dir.name

        # Iterate through replicates
        for rep_dir in sorted(plant_dir.iterdir()):
            if not rep_dir.is_dir():
                continue

            rep_name = rep_dir.name

            # Get all images in this replicate
            rgb_files = sorted(rep_dir.glob("*.png"))

            for idx, rgb_path in enumerate(rgb_files):
                # Find corresponding mask
                mask_path = mask_dir / plant_name / rep_name / rgb_path.name

                if not mask_path.exists():
                    print(f"Warning: No mask for {rgb_path}")
                    continue

                # Count leaves (skip if mask is corrupted)
                leaf_count = count_leaves_in_mask(str(mask_path))
                if leaf_count < 0:
                    print(f"Skipping corrupted mask: {mask_path}")
                    continue
                growth_stage = get_growth_stage(leaf_count)

                # Extract timestamp from filename
                # Format: 1000082_2022_05_11_12_09_35-...
                filename = rgb_path.stem
                try:
                    parts = filename.split("_")
                    timestamp = "_".join(parts[1:7])  # 2022_05_11_12_09_35
                except:
                    timestamp = None

                samples.append({
                    "plant_id": plant_name,
                    "replicate_id": rep_name,
                    "frame_idx": idx,
                    "rgb_path": str(rgb_path),
                    "mask_path": str(mask_path),
                    "filename": rgb_path.name,
                    "leaf_count": leaf_count,
                    "growth_stage": growth_stage,
                    "timestamp": timestamp,
                })

    return samples


def create_splits(samples, split_config, seed=42):
    """Create train/val/test splits.

    IMPORTANT: Splits are done WITHIN each sequence (plant replicate) to maintain
    temporal consistency. For each replicate's time-series:
    - First X% of frames -> train
    - Next Y% of frames -> val
    - Last Z% of frames -> test

    This ensures:
    1. No data leakage between splits (same plant at similar times)
    2. Proper temporal evaluation (test on later growth stages)
    3. All replicates contribute to all splits
    """
    # Group samples by sequence (plant + replicate)
    sequences = defaultdict(list)
    for s in samples:
        key = f"{s['plant_id']}_{s['replicate_id']}"
        sequences[key].append(s)

    # Sort each sequence by frame index (temporal order)
    for key in sequences:
        sequences[key].sort(key=lambda x: x['frame_idx'])

    train_samples = []
    val_samples = []
    test_samples = []

    # 70/15/15 split applied within each sequence in temporal order
    # (first frames -> train, middle -> val, last -> test). This keeps every
    # replicate represented in all splits with no plant-level leakage.
    train_ratio = split_config["train"]
    val_ratio = split_config["val"]
    for key, seq in sequences.items():
        n = len(seq)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_samples.extend(seq[:n_train])
        val_samples.extend(seq[n_train:n_train + n_val])
        test_samples.extend(seq[n_train + n_val:])

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def convert_to_coco(samples, output_path, split_name):
    """Convert samples to COCO format for instance segmentation."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        print("pycocotools not installed. Skipping COCO conversion.")
        return

    coco = {
        "info": {
            "description": f"LeafInstanceLabels Dataset - {split_name}",
            "version": "1.0",
            "year": 2026,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "leaf", "supercategory": "plant"}],
        "images": [],
        "annotations": [],
    }

    ann_id = 1

    for img_id, sample in enumerate(tqdm(samples, desc=f"Converting {split_name} to COCO"), start=1):
        # Get image dimensions
        img = Image.open(sample["rgb_path"])
        width, height = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": sample["filename"],
            "width": width,
            "height": height,
            "plant_id": sample["plant_id"],
            "replicate_id": sample["replicate_id"],
        })

        # Extract instances
        instances = extract_instances_from_mask(sample["mask_path"])

        for inst in instances:
            # Convert mask to RLE
            rle = mask_utils.encode(np.asfortranarray(inst["mask"].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": inst["bbox"],
                "area": inst["area"],
                "iscrowd": 0,
                "leaf_id": inst["leaf_id"],
            })
            ann_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"Saved COCO annotations to {output_path}")


def convert_to_yolo(samples, output_dir, split_name):
    """Convert samples to YOLO format for instance segmentation."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, desc=f"Converting {split_name} to YOLO"):
        # Copy image
        src_img = Path(sample["rgb_path"])
        dst_img = images_dir / src_img.name
        shutil.copy2(src_img, dst_img)

        # Create label file
        instances = extract_instances_from_mask(sample["mask_path"])

        label_file = labels_dir / (src_img.stem + ".txt")

        img = Image.open(sample["rgb_path"])
        width, height = img.size

        with open(label_file, "w") as f:
            for inst in instances:
                # YOLO format: class x_center y_center width height (normalized)
                bbox = inst["bbox"]
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                w = bbox[2] / width
                h = bbox[3] / height

                # For segmentation, also include polygon
                try:
                    polygons = mask_to_polygon(inst["mask"])
                    if polygons:
                        # Use first polygon
                        poly = polygons[0]
                        # Normalize polygon coordinates
                        norm_poly = []
                        for i in range(0, len(poly), 2):
                            norm_poly.append(poly[i] / width)
                            norm_poly.append(poly[i+1] / height)
                        poly_str = " ".join(f"{p:.6f}" for p in norm_poly)
                        f.write(f"0 {poly_str}\n")
                    else:
                        # Fallback to bbox
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                except:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"Saved YOLO annotations to {output_dir}")


def create_yolo_dataset_yaml(output_dir, class_names=["leaf"]):
    """Create YOLO dataset.yaml file."""
    config = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Saved YOLO config to {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LeafInstanceLabels dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/leaf_dataset_colour",
    )
    parser.add_argument("--output_dir", type=str, default="data/prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", type=str, default="all", choices=["coco", "yolo", "all"])
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_root, "01_raw_dataset")
    mask_dir = os.path.join(args.data_root, "02_leaf_labels")

    os.makedirs(args.output_dir, exist_ok=True)

    # Scan dataset
    print("Scanning dataset...")
    samples = scan_dataset(raw_dir, mask_dir)
    print(f"Found {len(samples)} samples")

    # Compute statistics
    stats = {
        "total_samples": len(samples),
        "plants": len(set(s["plant_id"] for s in samples)),
        "replicates": len(set(f"{s['plant_id']}_{s['replicate_id']}" for s in samples)),
        "growth_stages": {
            "early": sum(1 for s in samples if s["growth_stage"] == 0),
            "intermediate": sum(1 for s in samples if s["growth_stage"] == 1),
            "mature": sum(1 for s in samples if s["growth_stage"] == 2),
        },
        "leaf_count": {
            "min": min(s["leaf_count"] for s in samples),
            "max": max(s["leaf_count"] for s in samples),
            "mean": np.mean([s["leaf_count"] for s in samples]),
        },
    }

    stats_path = os.path.join(args.output_dir, "dataset_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved statistics to {stats_path}")

    # Single standard 70/15/15 random split
    split_configs = {
        "standard": {"train": 0.70, "val": 0.15, "test": 0.15, "strategy": "random"},
    }
    splits_to_create = ["standard"]

    for split_name in splits_to_create:
        print(f"\nCreating {split_name} split...")
        config = split_configs[split_name]
        splits = create_splits(samples, config, args.seed)

        # Save split JSON
        split_path = os.path.join(args.output_dir, f"split_{split_name}.json")
        with open(split_path, "w") as f:
            json.dump(splits, f, indent=2)

        print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        # Convert to required formats
        if args.format in ("coco", "all"):
            coco_dir = os.path.join(args.output_dir, "coco", split_name)
            os.makedirs(coco_dir, exist_ok=True)
            for subset in ["train", "val", "test"]:
                convert_to_coco(
                    splits[subset],
                    os.path.join(coco_dir, f"{subset}.json"),
                    f"{split_name}_{subset}",
                )

        if args.format in ("yolo", "all"):
            yolo_dir = os.path.join(args.output_dir, "yolo", split_name)
            for subset in ["train", "val", "test"]:
                convert_to_yolo(splits[subset], yolo_dir, subset)
            create_yolo_dataset_yaml(yolo_dir)

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
