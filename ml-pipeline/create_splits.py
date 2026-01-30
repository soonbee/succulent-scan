"""
Create train/val/test splits with group-aware stratified splitting.
Run this script once before training.
"""
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import get_config
from utils import (
    load_json,
    save_json,
    scan_dataset,
    set_seed,
    setup_logging,
)


def stratified_group_split(
    id_to_images: Dict[str, List[Dict]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split IDs into train/val/test sets with stratification by class.

    Strategy:
    1. Group IDs by their class (each ID belongs to one class)
    2. For each class, shuffle and split IDs according to ratios
    3. This ensures class distribution is maintained across splits

    Args:
        id_to_images: Dict mapping ID to list of image info
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    np.random.seed(seed)

    # Group IDs by class
    class_to_ids: Dict[str, List[str]] = defaultdict(list)
    for image_id, images in id_to_images.items():
        # All images of same ID should have same class
        class_name = images[0]["class_name"]
        class_to_ids[class_name].append(image_id)

    train_ids = []
    val_ids = []
    test_ids = []

    # Split each class separately to maintain distribution
    for class_name, ids in class_to_ids.items():
        ids = np.array(ids)
        np.random.shuffle(ids)

        n_total = len(ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_ids.extend(ids[:n_train].tolist())
        val_ids.extend(ids[n_train:n_train + n_val].tolist())
        test_ids.extend(ids[n_train + n_val:].tolist())

    return train_ids, val_ids, test_ids


def print_split_statistics(
    id_to_images: Dict[str, List[Dict]],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    logger
) -> None:
    """Print statistics about the splits"""

    def count_by_class(ids: List[str]) -> Dict[str, Tuple[int, int]]:
        """Count IDs and images per class"""
        class_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for image_id in ids:
            images = id_to_images[image_id]
            class_name = images[0]["class_name"]
            class_counts[class_name][0] += 1  # ID count
            class_counts[class_name][1] += len(images)  # Image count
        return {k: tuple(v) for k, v in class_counts.items()}

    train_counts = count_by_class(train_ids)
    val_counts = count_by_class(val_ids)
    test_counts = count_by_class(test_ids)

    logger.info("=" * 60)
    logger.info("Split Statistics")
    logger.info("=" * 60)

    # Overall statistics
    total_train_images = sum(len(id_to_images[i]) for i in train_ids)
    total_val_images = sum(len(id_to_images[i]) for i in val_ids)
    total_test_images = sum(len(id_to_images[i]) for i in test_ids)
    total_images = total_train_images + total_val_images + total_test_images

    logger.info(f"\nOverall:")
    logger.info(f"  Train: {len(train_ids):,} IDs, {total_train_images:,} images ({total_train_images/total_images*100:.1f}%)")
    logger.info(f"  Val:   {len(val_ids):,} IDs, {total_val_images:,} images ({total_val_images/total_images*100:.1f}%)")
    logger.info(f"  Test:  {len(test_ids):,} IDs, {total_test_images:,} images ({total_test_images/total_images*100:.1f}%)")
    logger.info(f"  Total: {len(train_ids) + len(val_ids) + len(test_ids):,} IDs, {total_images:,} images")

    # Per-class statistics
    logger.info(f"\nPer-class distribution:")
    logger.info(f"{'Class':<15} {'Train IDs':>10} {'Train Imgs':>12} {'Val IDs':>10} {'Val Imgs':>10} {'Test IDs':>10} {'Test Imgs':>10}")
    logger.info("-" * 85)

    all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
    for class_name in all_classes:
        train_id, train_img = train_counts.get(class_name, (0, 0))
        val_id, val_img = val_counts.get(class_name, (0, 0))
        test_id, test_img = test_counts.get(class_name, (0, 0))
        logger.info(f"{class_name:<15} {train_id:>10,} {train_img:>12,} {val_id:>10,} {val_img:>10,} {test_id:>10,} {test_img:>10,}")


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for split files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Load config
    config = get_config()

    # Override with command line arguments
    if args.data_dir:
        config.data.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.data.splits_dir = Path(args.output_dir)
    if args.seed:
        config.data.random_seed = args.seed

    # Setup
    logger = setup_logging()
    set_seed(config.data.random_seed)

    # Ensure output directory exists
    config.data.splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning dataset: {config.data.data_dir}")
    logger.info(f"Minimum image size: {config.data.min_image_size}x{config.data.min_image_size}")
    if config.data.class_names:
        logger.info(f"Using specified classes: {config.data.class_names}")
    else:
        logger.info("Auto-discovering classes from data directory")

    # Scan dataset
    id_to_images, class_to_idx, excluded_ids = scan_dataset(
        data_dir=config.data.data_dir,
        min_image_size=config.data.min_image_size,
        class_names=config.data.class_names,
    )

    logger.info(f"Found {len(id_to_images):,} valid IDs")
    if excluded_ids:
        logger.info(f"Excluded {len(excluded_ids):,} IDs due to small images")

    # Create splits
    logger.info(f"\nCreating splits with ratios: train={config.data.train_ratio}, "
                f"val={config.data.val_ratio}, test={config.data.test_ratio}")

    train_ids, val_ids, test_ids = stratified_group_split(
        id_to_images=id_to_images,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.random_seed
    )

    # Print statistics
    print_split_statistics(id_to_images, train_ids, val_ids, test_ids, logger)

    # Save splits
    save_json(train_ids, config.data.splits_dir / "train_ids.json")
    save_json(val_ids, config.data.splits_dir / "val_ids.json")
    save_json(test_ids, config.data.splits_dir / "test_ids.json")

    # Save class_to_idx
    save_json(class_to_idx, config.data.splits_dir / "class_to_idx.json")

    # Save metadata
    metadata = {
        "data_dir": str(config.data.data_dir),
        "min_image_size": config.data.min_image_size,
        "random_seed": config.data.random_seed,
        "train_ratio": config.data.train_ratio,
        "val_ratio": config.data.val_ratio,
        "test_ratio": config.data.test_ratio,
        "total_ids": len(id_to_images),
        "excluded_ids_count": len(excluded_ids),
        "train_ids_count": len(train_ids),
        "val_ids_count": len(val_ids),
        "test_ids_count": len(test_ids),
    }
    save_json(metadata, config.data.splits_dir / "split_metadata.json")

    logger.info(f"\nSplit files saved to: {config.data.splits_dir}")
    logger.info("  - train_ids.json")
    logger.info("  - val_ids.json")
    logger.info("  - test_ids.json")
    logger.info("  - class_to_idx.json")
    logger.info("  - split_metadata.json")


if __name__ == "__main__":
    main()
