"""
Dataset and Sampler for Image Classification
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from config import AugmentationConfig, DataConfig
from utils import extract_id_from_filename, load_json


class ImageDataset(Dataset):
    """Dataset for image classification"""

    def __init__(
        self,
        data_dir: Path,
        id_list: List[str],
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_dir: Root directory containing class subdirectories
            id_list: List of IDs to include in this dataset
            class_to_idx: Mapping from class name to index
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.id_set = set(id_list)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

        # Collect all images for the given IDs
        self.samples: List[Dict] = []
        self.class_to_samples: Dict[int, List[int]] = defaultdict(list)

        self._scan_images()

    def _scan_images(self):
        """Scan data directory and collect image samples"""
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            for ext in image_extensions:
                for image_path in class_dir.glob(ext):
                    image_id = extract_id_from_filename(image_path.name)
                    if image_id in self.id_set:
                        sample_idx = len(self.samples)
                        self.samples.append({
                            "path": image_path,
                            "class_idx": class_idx,
                            "class_name": class_name,
                            "id": image_id,
                        })
                        self.class_to_samples[class_idx].append(sample_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, sample["class_idx"]

    def get_labels(self) -> List[int]:
        """Get list of all labels for sampler"""
        return [s["class_idx"] for s in self.samples]


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler that creates batches with P classes and K samples per class.

    Each batch contains exactly P*K samples, with K samples from each of P classes.
    This is optimal for metric learning methods like ArcFace.
    """

    def __init__(
        self,
        dataset: ImageDataset,
        num_classes_per_batch: Optional[int] = None,  # P: None means all classes
        samples_per_class: int = 4,  # K
        drop_last: bool = True,
    ):
        """
        Args:
            dataset: ImageDataset instance
            num_classes_per_batch: Number of classes per batch (P). None means all classes.
            samples_per_class: Number of samples per class per batch (K)
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last

        # Get class to sample indices mapping
        self.class_to_samples = dataset.class_to_samples
        self.num_classes = len(self.class_to_samples)

        # If num_classes_per_batch is None, use all classes
        self.num_classes_per_batch = num_classes_per_batch if num_classes_per_batch else self.num_classes

        self.batch_size = self.num_classes_per_batch * samples_per_class

        # Verify we have enough classes
        if self.num_classes < self.num_classes_per_batch:
            raise ValueError(
                f"Dataset has {self.num_classes} classes, but num_classes_per_batch is {self.num_classes_per_batch}"
            )

        # Calculate number of batches
        self._calculate_num_batches()

    def _calculate_num_batches(self):
        """Calculate the number of batches based on the smallest class"""
        # For P×K sampling, we need to consider how many complete batches we can create
        # Each class should contribute K samples per batch where it's selected

        # Calculate total samples needed if all classes are in every batch
        min_class_samples = min(len(samples) for samples in self.class_to_samples.values())

        # Number of times we can sample K items from the smallest class
        max_iterations_per_class = min_class_samples // self.samples_per_class

        # Total batches possible
        self.num_batches = max_iterations_per_class

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches"""
        # Create a copy of sample indices per class for sampling
        class_samples = {
            cls: np.array(samples.copy())
            for cls, samples in self.class_to_samples.items()
        }

        # Shuffle samples within each class
        for cls in class_samples:
            np.random.shuffle(class_samples[cls])

        # Track position in each class
        class_positions = {cls: 0 for cls in class_samples}

        # Get list of class indices
        class_indices = list(class_samples.keys())

        for _ in range(self.num_batches):
            batch = []

            # Select P classes for this batch
            # If P equals total classes, use all; otherwise randomly select
            if self.num_classes_per_batch >= self.num_classes:
                selected_classes = class_indices
            else:
                selected_classes = np.random.choice(
                    class_indices, self.num_classes_per_batch, replace=False
                ).tolist()

            # Select K samples from each selected class
            for cls in selected_classes:
                samples = class_samples[cls]
                pos = class_positions[cls]

                # If we've exhausted this class, reshuffle and reset
                if pos + self.samples_per_class > len(samples):
                    np.random.shuffle(class_samples[cls])
                    class_positions[cls] = 0
                    pos = 0

                # Get K samples
                batch.extend(samples[pos:pos + self.samples_per_class].tolist())
                class_positions[cls] = pos + self.samples_per_class

            yield batch

    def __len__(self) -> int:
        return self.num_batches


def get_train_transform(config: AugmentationConfig, image_size: int) -> transforms.Compose:
    """Get training data augmentation transforms"""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(config.crop_scale_min, config.crop_scale_max),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
        transforms.RandomRotation(config.rotation_degrees),
        transforms.ColorJitter(
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


def get_val_transform(config: AugmentationConfig, image_size: int) -> transforms.Compose:
    """
    Get validation/test transforms.
    Resize shorter side to image_size (maintain aspect ratio) -> CenterCrop -> Normalize
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


def create_dataloaders(
    data_config: DataConfig,
    aug_config: AugmentationConfig,
    training_config,
    splits_dir: Optional[Path] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int]]:
    """
    Create train and validation dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, class_to_idx)
    """
    splits_dir = splits_dir or data_config.splits_dir

    # Load splits
    train_ids = load_json(splits_dir / "train_ids.json")
    val_ids = load_json(splits_dir / "val_ids.json")
    class_to_idx = load_json(splits_dir / "class_to_idx.json")

    # Create transforms
    train_transform = get_train_transform(aug_config, data_config.image_size)
    val_transform = get_val_transform(aug_config, data_config.image_size)

    # Create datasets
    train_dataset = ImageDataset(
        data_dir=data_config.data_dir,
        id_list=train_ids,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )

    val_dataset = ImageDataset(
        data_dir=data_config.data_dir,
        id_list=val_ids,
        class_to_idx=class_to_idx,
        transform=val_transform,
    )

    # Create balanced batch sampler for training
    train_sampler = BalancedBatchSampler(
        dataset=train_dataset,
        num_classes_per_batch=training_config.num_classes_per_batch,
        samples_per_class=training_config.samples_per_class,
    )

    # Create dataloaders (pin_memory only for CUDA)
    use_pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=training_config.num_workers,
        pin_memory=use_pin_memory,
    )

    # Calculate validation batch size (use same as train sampler's effective batch size)
    val_batch_size = train_sampler.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, class_to_idx
