"""
Utility functions for training pipeline
"""
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("image_classifier")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load data from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions (width, height) without loading full image"""
    with Image.open(image_path) as img:
        return img.size


def check_image_valid(image_path: Path, min_size: int = 224) -> bool:
    """Check if image meets minimum size requirement"""
    try:
        width, height = get_image_size(image_path)
        return width >= min_size and height >= min_size
    except Exception:
        return False


def extract_id_from_filename(filename: str) -> str:
    """Extract ID from filename (ID_sequence.jpg -> ID)"""
    stem = Path(filename).stem  # Remove extension
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0]
    return stem


def scan_dataset(
    data_dir: Path,
    min_image_size: int = 224,
    class_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, int], List[str]]:
    """
    Scan dataset directory and collect image information.

    Args:
        data_dir: Root directory containing class subdirectories
        min_image_size: Minimum image size threshold
        class_names: List of class names to use. If None, auto-discover from subdirectories.

    Returns:
        - id_to_images: Dict mapping ID to list of image info dicts
        - class_to_idx: Dict mapping class name to index
        - excluded_ids: List of IDs excluded due to small images
    """
    id_to_images: Dict[str, List[Dict]] = {}
    excluded_ids: List[str] = []

    # Use provided class names or auto-discover from subdirectories
    if class_names is None:
        class_names = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    else:
        # Validate that specified class directories exist
        missing = [c for c in class_names if not (data_dir / c).exists()]
        if missing:
            raise ValueError(f"Class directories not found: {missing}")
        class_names = sorted(class_names)

    if not class_names:
        raise ValueError(f"No class directories found in {data_dir}")

    # Create class_to_idx mapping (alphabetically sorted)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = data_dir / class_name
        class_idx = class_to_idx[class_name]

        # Support multiple image formats
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        for ext in image_extensions:
            for image_path in class_dir.glob(ext):
                image_id = extract_id_from_filename(image_path.name)

                if image_id not in id_to_images:
                    id_to_images[image_id] = []

                id_to_images[image_id].append({
                    "path": str(image_path),
                    "class_name": class_name,
                    "class_idx": class_idx,
                    "id": image_id,
                })

    # Check for small images and exclude entire ID groups
    ids_to_remove = []
    for image_id, images in id_to_images.items():
        for img_info in images:
            if not check_image_valid(Path(img_info["path"]), min_image_size):
                ids_to_remove.append(image_id)
                break

    for image_id in ids_to_remove:
        excluded_ids.append(image_id)
        del id_to_images[image_id]

    return id_to_images, class_to_idx, excluded_ids


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if this is the best score, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Path,
    is_best: bool = False,
    best_path: Optional[Path] = None
) -> None:
    """Save checkpoint to file"""
    torch.save(state, filepath)
    if is_best and best_path is not None:
        torch.save(state, best_path)


def load_checkpoint(filepath: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None) -> Dict[str, Any]:
    """Load checkpoint from file"""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
