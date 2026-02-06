"""
Build Faiss index from trained model for inference server.

Scans the entire dataset (no train/val/test split), extracts embeddings
using a trained checkpoint, and saves a Faiss index + metadata to disk.

Usage:
    python build_index.py --data_dir ./data --checkpoint ./checkpoints/best_model.pt
"""
import os
# Prevent OpenMP conflicts between PyTorch and Faiss on non-CUDA systems
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset import ImageDataset, get_val_transform
from evaluate import build_faiss_index, extract_embeddings
from model import create_model
from utils import save_json, scan_dataset, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Faiss index for inference server"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./index", help="Output directory for index files"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for embedding extraction"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    config = get_config()

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1. Scan entire dataset
    logger.info(f"Scanning dataset: {data_dir}")
    id_to_images, class_to_idx, excluded_ids = scan_dataset(
        data_dir, min_image_size=config.data.min_image_size
    )

    all_ids = sorted(id_to_images.keys())
    num_classes = len(class_to_idx)

    logger.info(f"Found {len(all_ids)} IDs across {num_classes} classes")
    if excluded_ids:
        logger.info(f"Excluded {len(excluded_ids)} IDs due to small images")

    # 2. Load model from checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Infer num_classes and embedding_dim from checkpoint
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        ckpt_num_classes = checkpoint_config.get("num_classes", num_classes)
        ckpt_embedding_dim = checkpoint_config.get("embedding_dim", config.model.embedding_dim)
    else:
        ckpt_num_classes = num_classes
        ckpt_embedding_dim = config.model.embedding_dim

    device = config.training.device
    model = create_model(
        num_classes=ckpt_num_classes,
        embedding_dim=ckpt_embedding_dim,
        pretrained=False,
        arcface_s=config.model.arcface_s,
        arcface_m=config.model.arcface_m_finetune,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # 3. Create dataset with validation transform (deterministic, no augmentation)
    transform = get_val_transform(config.augmentation, config.data.image_size)

    dataset = ImageDataset(
        data_dir=data_dir,
        id_list=all_ids,
        class_to_idx=class_to_idx,
        transform=transform,
    )
    logger.info(f"Gallery dataset: {len(dataset)} images")

    use_pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=use_pin_memory,
    )

    # 4. Extract embeddings
    logger.info("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, dataloader, device)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # 5. Build Faiss index
    logger.info("Building Faiss index...")
    index = build_faiss_index(embeddings)
    logger.info(f"Faiss index: {index.ntotal} vectors, {index.d} dimensions")

    # 6. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "gallery.index"
    labels_path = output_dir / "gallery_labels.npy"
    class_to_idx_path = output_dir / "class_to_idx.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"Saved Faiss index: {index_path}")

    np.save(str(labels_path), labels)
    logger.info(f"Saved gallery labels: {labels_path}")

    save_json(class_to_idx, class_to_idx_path)
    logger.info(f"Saved class mapping: {class_to_idx_path}")

    logger.info("Index build complete!")


if __name__ == "__main__":
    main()
