"""
Evaluation module using Faiss for retrieval-based metrics.
"""
import os
# Prevent OpenMP conflicts between PyTorch and Faiss on non-CUDA systems
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from dataset import ImageDataset, get_val_transform
from model import EmbeddingModel, create_model
from utils import load_json, setup_logging


def extract_embeddings(
    model: EmbeddingModel,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from a dataset.

    Args:
        model: The embedding model
        dataloader: DataLoader for the dataset
        device: Device to use

    Returns:
        Tuple of (embeddings, labels) as numpy arrays
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings = model.get_embeddings(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    labels = np.concatenate(all_labels)

    return embeddings, labels


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build Faiss index for L2 distance search.

    Args:
        embeddings: Gallery embeddings (N, D)

    Returns:
        Faiss index
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_faiss(
    index: faiss.IndexFlatL2,
    query_embeddings: np.ndarray,
    k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for nearest neighbors using Faiss.

    Args:
        index: Faiss index
        query_embeddings: Query embeddings (Q, D)
        k: Number of neighbors to retrieve

    Returns:
        Tuple of (distances, indices)
    """
    distances, indices = index.search(query_embeddings, k)
    return distances, indices


def compute_recall_at_k(
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    retrieved_indices: np.ndarray,
    k_values: List[int] = [1, 5]
) -> Dict[str, float]:
    """
    Compute Recall@K metrics.

    Recall@K: Success if at least one of the top-K retrieved items has the same label.

    Args:
        query_labels: Labels of query samples (Q,)
        gallery_labels: Labels of gallery samples (G,)
        retrieved_indices: Indices of retrieved samples (Q, max_k)
        k_values: List of K values to compute recall for

    Returns:
        Dictionary mapping "recall_at_k" to recall value
    """
    results = {}
    num_queries = len(query_labels)

    for k in k_values:
        correct = 0
        for i in range(num_queries):
            query_label = query_labels[i]
            retrieved_labels = gallery_labels[retrieved_indices[i, :k]]
            if query_label in retrieved_labels:
                correct += 1
        results[f"recall_at_{k}"] = correct / num_queries

    return results


def compute_map(
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    retrieved_indices: np.ndarray,
    k: int = 20
) -> float:
    """
    Compute Mean Average Precision (mAP).

    Args:
        query_labels: Labels of query samples (Q,)
        gallery_labels: Labels of gallery samples (G,)
        retrieved_indices: Indices of retrieved samples (Q, k)
        k: Number of retrieved items to consider

    Returns:
        mAP score
    """
    num_queries = len(query_labels)
    average_precisions = []

    for i in range(num_queries):
        query_label = query_labels[i]
        retrieved_labels = gallery_labels[retrieved_indices[i, :k]]

        # Compute precision at each position where we find a relevant item
        relevant = (retrieved_labels == query_label).astype(float)
        if relevant.sum() == 0:
            average_precisions.append(0.0)
            continue

        # Cumulative sum of relevant items
        cum_relevant = np.cumsum(relevant)
        # Precision at each position
        precisions = cum_relevant / np.arange(1, k + 1)
        # Average precision: mean of precisions at relevant positions
        ap = (precisions * relevant).sum() / relevant.sum()
        average_precisions.append(ap)

    return np.mean(average_precisions)


def compute_confusion_matrix(
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    retrieved_indices: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix based on top-1 retrieval.

    Args:
        query_labels: Labels of query samples (Q,)
        gallery_labels: Labels of gallery samples (G,)
        retrieved_indices: Indices of retrieved samples (Q, k)
        num_classes: Number of classes

    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(len(query_labels)):
        true_label = query_labels[i]
        pred_label = gallery_labels[retrieved_indices[i, 0]]  # Top-1
        confusion[true_label, pred_label] += 1

    return confusion


def print_confusion_matrix(
    confusion: np.ndarray,
    class_names: List[str],
    logger
) -> None:
    """Print formatted confusion matrix"""
    logger.info("\nConfusion Matrix (rows=true, cols=predicted):")

    # Header
    header = "           " + "".join(f"{name[:8]:>10}" for name in class_names)
    logger.info(header)
    logger.info("-" * len(header))

    # Rows
    for i, name in enumerate(class_names):
        row = f"{name[:10]:<10}" + "".join(f"{confusion[i, j]:>10}" for j in range(len(class_names)))
        logger.info(row)


def evaluate_model(
    model: EmbeddingModel,
    gallery_loader: DataLoader,
    query_loader: DataLoader,
    class_names: List[str],
    k_values: List[int] = [1, 5],
    top_k: int = 20,
    device: str = "cuda",
    logger=None
) -> Dict[str, float]:
    """
    Evaluate model using retrieval-based metrics.

    Args:
        model: The embedding model
        gallery_loader: DataLoader for gallery (train) set
        query_loader: DataLoader for query (val/test) set
        class_names: List of class names
        k_values: K values for Recall@K computation
        top_k: Number of neighbors to retrieve
        device: Device to use
        logger: Logger instance

    Returns:
        Dictionary of evaluation metrics
    """
    if logger is None:
        logger = setup_logging()

    logger.info("Extracting gallery embeddings...")
    gallery_embeddings, gallery_labels = extract_embeddings(model, gallery_loader, device)
    logger.info(f"Gallery: {len(gallery_labels)} samples")

    logger.info("Extracting query embeddings...")
    query_embeddings, query_labels = extract_embeddings(model, query_loader, device)
    logger.info(f"Query: {len(query_labels)} samples")

    logger.info("Building Faiss index...")
    index = build_faiss_index(gallery_embeddings)

    logger.info("Searching...")
    distances, indices = search_faiss(index, query_embeddings, k=top_k)

    # Compute metrics
    logger.info("Computing metrics...")

    recall_results = compute_recall_at_k(
        query_labels, gallery_labels, indices, k_values
    )

    map_score = compute_map(
        query_labels, gallery_labels, indices, k=top_k
    )

    confusion = compute_confusion_matrix(
        query_labels, gallery_labels, indices, len(class_names)
    )

    # Log results
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)

    for k in k_values:
        logger.info(f"Recall@{k}: {recall_results[f'recall_at_{k}']:.4f}")

    logger.info(f"mAP@{top_k}: {map_score:.4f}")

    print_confusion_matrix(confusion, class_names, logger)

    # Compile results
    results = {
        **recall_results,
        f"map_at_{top_k}": map_score,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--splits_dir", type=str, default=None, help="Splits directory")
    args = parser.parse_args()

    # Load config
    config = get_config()
    logger = setup_logging()

    if args.data_dir:
        config.data.data_dir = Path(args.data_dir)
    if args.splits_dir:
        config.data.splits_dir = Path(args.splits_dir)

    # Load class mapping
    class_to_idx = load_json(config.data.splits_dir / "class_to_idx.json")
    class_names = sorted(class_to_idx.keys())
    num_classes = len(class_names)

    # Load splits
    train_ids = load_json(config.data.splits_dir / "train_ids.json")

    if args.split == "val":
        query_ids = load_json(config.data.splits_dir / "val_ids.json")
    else:
        query_ids = load_json(config.data.splits_dir / "test_ids.json")

    logger.info(f"Evaluating on {args.split} set")

    # Create transforms
    transform = get_val_transform(config.augmentation, config.data.image_size)

    # Create datasets
    gallery_dataset = ImageDataset(
        data_dir=config.data.data_dir,
        id_list=train_ids,
        class_to_idx=class_to_idx,
        transform=transform,
    )

    query_dataset = ImageDataset(
        data_dir=config.data.data_dir,
        id_list=query_ids,
        class_to_idx=class_to_idx,
        transform=transform,
    )

    # Create dataloaders
    p = config.training.num_classes_per_batch or num_classes
    batch_size = p * config.training.samples_per_class

    use_pin_memory = torch.cuda.is_available()

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=use_pin_memory,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=use_pin_memory,
    )

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = create_model(
        num_classes=num_classes,
        embedding_dim=config.model.embedding_dim,
        pretrained=False,  # We'll load weights from checkpoint
        arcface_s=config.model.arcface_s,
        arcface_m=config.model.arcface_m_finetune,
        device=config.training.device,
    )

    checkpoint = torch.load(args.checkpoint, map_location=config.training.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Evaluate
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

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
