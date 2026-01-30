"""
Configuration for Image Classification Training
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch


def get_device() -> str:
    """Auto-detect available device: CUDA if available, otherwise CPU"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_gpu_capabilities() -> dict:
    """
    Detect GPU architecture and return optimal settings.

    Returns:
        dict with keys:
            - name: GPU name (e.g., "NVIDIA A100-SXM4-80GB")
            - compute_capability: tuple (major, minor)
            - supports_bf16: bool (Ampere+ architecture)
            - vram_gb: float
            - recommended_workers: int
            - recommended_accumulation_steps: int
    """
    if not torch.cuda.is_available():
        return {
            "name": "CPU",
            "compute_capability": (0, 0),
            "supports_bf16": False,
            "vram_gb": 0,
            "recommended_workers": 4,
            "recommended_accumulation_steps": 4,
        }

    props = torch.cuda.get_device_properties(0)
    compute_cap = (props.major, props.minor)
    vram_gb = props.total_memory / (1024**3)

    # Ampere (8.0+) supports BF16 natively
    # A100: 8.0, RTX 3090: 8.6, H100: 9.0
    supports_bf16 = compute_cap >= (8, 0)

    # Adjust settings based on VRAM
    if vram_gb >= 70:  # A100 80GB, H100
        recommended_workers = 16
        recommended_accumulation = 1
    elif vram_gb >= 35:  # A100 40GB
        recommended_workers = 8
        recommended_accumulation = 2
    else:  # RTX 3090 24GB, etc.
        recommended_workers = 4
        recommended_accumulation = 4

    return {
        "name": props.name,
        "compute_capability": compute_cap,
        "supports_bf16": supports_bf16,
        "vram_gb": vram_gb,
        "recommended_workers": recommended_workers,
        "recommended_accumulation_steps": recommended_accumulation,
    }


@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: Path = Path("./data")
    splits_dir: Path = Path("./splits")

    # Image settings
    image_size: int = 480
    min_image_size: int = 224  # Exclude images below this threshold

    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Class names: None for auto-discovery, or specify list for explicit control
    class_names: Optional[List[str]] = None

    random_seed: int = 42


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    # RandomResizedCrop
    crop_scale_min: float = 0.8
    crop_scale_max: float = 1.0

    # RandomHorizontalFlip
    horizontal_flip_prob: float = 0.5

    # RandomRotation
    rotation_degrees: int = 15

    # ColorJitter (conservative for plant color preservation)
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    hue: float = 0.05

    # ImageNet normalization
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    backbone: str = "efficientnet_v2_l"
    embedding_dim: int = 512
    pretrained: bool = True

    # ArcFace parameters
    arcface_s: float = 64.0  # Scale
    arcface_m_warmup: float = 0.0  # Margin for warm-up phase
    arcface_m_finetune: float = 0.5  # Margin for fine-tuning phase


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Phases
    warmup_epochs: int = 5
    total_epochs: int = 50

    # Batch settings (P×K format)
    num_classes_per_batch: Optional[int] = None  # P: None means all classes
    samples_per_class: int = 4  # K: samples per class
    # batch_size = P × K

    # Gradient accumulation (None = auto based on GPU)
    gradient_accumulation_steps: Optional[int] = None

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Learning rate scheduler
    scheduler_type: str = "cosine_warm_restarts"  # or "reduce_on_plateau"
    # CosineAnnealingWarmRestarts params
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    # ReduceLROnPlateau params
    plateau_patience: int = 5
    plateau_factor: float = 0.5

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "recall_at_1"

    # Mixed precision (None = auto: BF16 on Ampere+, FP16 otherwise)
    use_amp: bool = True
    use_bf16: Optional[bool] = None

    # Checkpointing
    checkpoint_dir: Path = Path("./checkpoints")
    save_every_n_epochs: int = 10

    # Device and workers (None = auto based on GPU)
    device: str = field(default_factory=get_device)
    num_workers: Optional[int] = None

    # Internal: resolved GPU capabilities (set in __post_init__)
    _gpu_info: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Auto-configure settings based on detected GPU"""
        self._gpu_info = get_gpu_capabilities()

        # Auto-select BF16 if supported
        if self.use_bf16 is None:
            self.use_bf16 = self._gpu_info["supports_bf16"]

        # Auto-select num_workers
        if self.num_workers is None:
            self.num_workers = self._gpu_info["recommended_workers"]

        # Auto-select gradient accumulation steps
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = self._gpu_info["recommended_accumulation_steps"]


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Faiss
    faiss_index_type: str = "IndexFlatL2"

    # Recall@K
    recall_k_values: List[int] = field(default_factory=lambda: [1, 5])

    # Top-K for retrieval
    top_k: int = 20


@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment name
    experiment_name: str = "image_classifier_v1"

    def __post_init__(self):
        """Ensure directories exist"""
        self.data.splits_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration"""
    return Config()
