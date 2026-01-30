"""
Main training script for Image Classification.
"""
import os
# Prevent OpenMP conflicts between PyTorch and Faiss on non-CUDA systems
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# AMP: Only use with CUDA, simple fallback for CPU
from contextlib import nullcontext

# CUDA optimizations
if torch.cuda.is_available():
    # Enable cudnn benchmark for fixed input sizes (faster convolutions)
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for Ampere+ GPUs (automatic speedup with minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def create_grad_scaler(enabled: bool):
        return torch.amp.GradScaler("cuda", enabled=enabled)
else:
    # Simple dummy class for CPU
    def create_grad_scaler(enabled: bool):
        class _DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        return _DummyScaler()


def get_autocast_context(device: str, enabled: bool, use_bf16: bool = False):
    """
    Get autocast context for mixed precision training.

    Args:
        device: "cuda" or "cpu"
        enabled: Whether AMP is enabled
        use_bf16: Use BFloat16 instead of Float16 (recommended for Ampere+ GPUs)

    Returns:
        Context manager for autocast
    """
    if enabled and device == "cuda":
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()

from config import Config, get_config
from dataset import (
    ImageDataset,
    BalancedBatchSampler,
    create_dataloaders,
    get_val_transform,
)
from evaluate import evaluate_model
from model import EmbeddingModel, create_model
from utils import (
    AverageMeter,
    EarlyStopping,
    load_json,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logging,
)


def train_one_epoch(
    model: EmbeddingModel,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Config,
    epoch: int,
    logger,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        config: Configuration
        epoch: Current epoch number
        logger: Logger instance

    Returns:
        Dictionary of training metrics
    """
    model.train()
    loss_meter = AverageMeter()

    accumulation_steps = config.training.gradient_accumulation_steps
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(config.training.device)
        labels = labels.to(config.training.device)

        # Mixed precision forward pass
        with get_autocast_context(config.training.device, config.training.use_amp, config.training.use_bf16):
            logits, _ = model(images, labels)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update metrics
        loss_meter.update(loss.item() * accumulation_steps, images.size(0))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    return {"train_loss": loss_meter.avg}


def validate(
    model: EmbeddingModel,
    gallery_loader: DataLoader,
    query_loader: DataLoader,
    class_names: list,
    config: Config,
    logger,
) -> Dict[str, float]:
    """
    Validate the model using retrieval-based metrics.

    Args:
        model: The model to validate
        gallery_loader: Gallery (train) data loader
        query_loader: Query (val) data loader
        class_names: List of class names
        config: Configuration
        logger: Logger instance

    Returns:
        Dictionary of validation metrics
    """
    results = evaluate_model(
        model=model,
        gallery_loader=gallery_loader,
        query_loader=query_loader,
        class_names=class_names,
        k_values=config.evaluation.recall_k_values,
        top_k=config.evaluation.top_k,
        device=config.training.device,
        logger=logger,
    )
    return results


def train(config: Config, logger, fresh: bool = False):
    """
    Main training function.

    Args:
        config: Configuration object
        logger: Logger instance
        fresh: If True, ignore existing checkpoints and start fresh
    """
    # Set seed for reproducibility
    set_seed(config.data.random_seed)

    # Load class mapping
    class_to_idx = load_json(config.data.splits_dir / "class_to_idx.json")
    class_names = sorted(class_to_idx.keys())
    num_classes = len(class_names)

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        data_config=config.data,
        aug_config=config.augmentation,
        training_config=config.training,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create gallery loader for validation (same data as train, but with val transform)
    train_ids = load_json(config.data.splits_dir / "train_ids.json")
    val_transform = get_val_transform(config.augmentation, config.data.image_size)

    gallery_dataset = ImageDataset(
        data_dir=config.data.data_dir,
        id_list=train_ids,
        class_to_idx=class_to_idx,
        transform=val_transform,
    )

    # num_classes_per_batch가 None이면 전체 클래스 수 사용
    p = config.training.num_classes_per_batch or num_classes
    gallery_batch_size = p * config.training.samples_per_class

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=gallery_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(
        num_classes=num_classes,
        embedding_dim=config.model.embedding_dim,
        pretrained=config.model.pretrained,
        arcface_s=config.model.arcface_s,
        arcface_m=config.model.arcface_m_warmup,  # Start with m=0 for warm-up
        device=config.training.device,
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Learning rate scheduler
    if config.training.scheduler_type == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.cosine_t0,
            T_mult=config.training.cosine_t_mult,
        )
    else:  # reduce_on_plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=config.training.plateau_patience,
            factor=config.training.plateau_factor,
        )

    # Mixed precision scaler (not needed for BF16 - it has sufficient dynamic range)
    use_grad_scaler = config.training.use_amp and not config.training.use_bf16
    scaler = create_grad_scaler(use_grad_scaler)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode="max"
    )

    # Training history
    history = {
        "train_loss": [],
        "val_recall_at_1": [],
        "val_recall_at_5": [],
        "val_map": [],
        "lr": [],
    }

    best_metric = 0.0
    start_epoch = 1

    # Check for checkpoint to resume
    latest_checkpoint = config.training.checkpoint_dir / "latest_model.pt"
    if latest_checkpoint.exists() and not fresh:
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=config.training.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", 0.0)
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, best_metric: {best_metric:.4f}")
    elif fresh and latest_checkpoint.exists():
        logger.info("Fresh start requested, ignoring existing checkpoints")

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, config.training.total_epochs + 1):
        # Phase management
        if epoch <= config.training.warmup_epochs:
            # Phase 1: Warm-up (backbone frozen, m=0)
            phase = "warmup"
            if epoch == 1 or (epoch == start_epoch and start_epoch <= config.training.warmup_epochs):
                logger.info(f"\n{'='*50}")
                logger.info("Phase 1: Warm-up (Backbone frozen, m=0)")
                logger.info(f"{'='*50}")
                model.freeze_backbone()
                model.set_arcface_margin(config.model.arcface_m_warmup)
        else:
            # Phase 2: Fine-tuning (backbone unfrozen, m=0.5)
            phase = "finetune"
            if epoch == config.training.warmup_epochs + 1:
                logger.info(f"\n{'='*50}")
                logger.info("Phase 2: Fine-tuning (Backbone unfrozen, m=0.5)")
                logger.info(f"{'='*50}")
                model.unfreeze_backbone()
                model.set_arcface_margin(config.model.arcface_m_finetune)

                # Reset optimizer for fine-tuning with potentially different learning rates
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.training.learning_rate,
                    weight_decay=config.training.weight_decay,
                )

                # Reset scheduler
                if config.training.scheduler_type == "cosine_warm_restarts":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=config.training.cosine_t0,
                        T_mult=config.training.cosine_t_mult,
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        patience=config.training.plateau_patience,
                        factor=config.training.plateau_factor,
                    )

        # Train one epoch
        logger.info(f"\nEpoch {epoch}/{config.training.total_epochs} [{phase}]")
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            epoch=epoch,
            logger=logger,
        )

        # Validate
        logger.info("Validating...")
        val_metrics = validate(
            model=model,
            gallery_loader=gallery_loader,
            query_loader=val_loader,
            class_names=class_names,
            config=config,
            logger=logger,
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update scheduler
        if config.training.scheduler_type == "cosine_warm_restarts":
            scheduler.step()
        else:
            scheduler.step(val_metrics["recall_at_1"])

        # Update history
        history["train_loss"].append(train_metrics["train_loss"])
        history["val_recall_at_1"].append(val_metrics["recall_at_1"])
        history["val_recall_at_5"].append(val_metrics["recall_at_5"])
        history["val_map"].append(val_metrics.get(f"map_at_{config.evaluation.top_k}", 0))
        history["lr"].append(current_lr)

        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Val Recall@1: {val_metrics['recall_at_1']:.4f}")
        logger.info(f"Val Recall@5: {val_metrics['recall_at_5']:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")

        # Check for best model
        current_metric = val_metrics["recall_at_1"]
        is_best = early_stopping(current_metric)

        if is_best:
            best_metric = current_metric
            logger.info(f"New best model! Recall@1: {best_metric:.4f}")

        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "config": {
                "embedding_dim": config.model.embedding_dim,
                "num_classes": num_classes,
            },
        }

        # Save latest
        save_checkpoint(
            checkpoint_state,
            config.training.checkpoint_dir / "latest_model.pt",
            is_best=is_best,
            best_path=config.training.checkpoint_dir / "best_model.pt"
        )

        # Save periodic checkpoint
        if epoch % config.training.save_every_n_epochs == 0:
            save_checkpoint(
                checkpoint_state,
                config.training.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            )
            logger.info(f"Saved checkpoint at epoch {epoch}")

        # Early stopping check
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Save training history
    save_json(history, config.training.checkpoint_dir / "training_history.json")

    logger.info("\nTraining complete!")
    logger.info(f"Best Recall@1: {best_metric:.4f}")
    logger.info(f"Checkpoints saved to: {config.training.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Image Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--splits_dir", type=str, default=None, help="Splits directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="Warm-up epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    # GPU optimization overrides
    parser.add_argument("--use_bf16", type=lambda x: x.lower() == "true", default=None,
                        help="Use BF16 instead of FP16 (auto-detected if not specified)")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--accum_steps", type=int, default=None, help="Gradient accumulation steps")
    # Training control
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh training, ignore existing checkpoints")
    args = parser.parse_args()

    # Load config (auto-detects GPU and sets optimal defaults)
    config = get_config()

    # Override with command line arguments
    if args.data_dir:
        config.data.data_dir = Path(args.data_dir)
    if args.splits_dir:
        config.data.splits_dir = Path(args.splits_dir)
    if args.checkpoint_dir:
        config.training.checkpoint_dir = Path(args.checkpoint_dir)
    if args.epochs:
        config.training.total_epochs = args.epochs
    if args.warmup_epochs:
        config.training.warmup_epochs = args.warmup_epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.seed:
        config.data.random_seed = args.seed
    # GPU optimization overrides
    if args.use_bf16 is not None:
        config.training.use_bf16 = args.use_bf16
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.accum_steps is not None:
        config.training.gradient_accumulation_steps = args.accum_steps

    # Ensure directories exist
    config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = config.training.checkpoint_dir / "training.log"
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Image Classification Training")
    logger.info("=" * 60)
    logger.info(f"Data directory: {config.data.data_dir}")
    logger.info(f"Splits directory: {config.data.splits_dir}")
    logger.info(f"Checkpoint directory: {config.training.checkpoint_dir}")

    # Log GPU info and auto-configured settings
    gpu_info = config.training._gpu_info
    logger.info(f"Device: {config.training.device}")
    if config.training.device == "cuda":
        logger.info(f"GPU: {gpu_info['name']}")
        logger.info(f"VRAM: {gpu_info['vram_gb']:.1f} GB")
        logger.info(f"Compute Capability: {gpu_info['compute_capability']}")
    logger.info(f"Mixed Precision: {'BF16' if config.training.use_bf16 else 'FP16'} (AMP={config.training.use_amp})")
    logger.info(f"Num Workers: {config.training.num_workers}")
    logger.info(f"Gradient Accumulation Steps: {config.training.gradient_accumulation_steps}")

    # Start training
    train(config, logger, fresh=args.fresh)


if __name__ == "__main__":
    main()
