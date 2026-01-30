"""
Model architecture: EfficientNet V2 Large + ArcFace
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) head for metric learning.

    ArcFace adds an angular margin penalty to improve intra-class compactness
    and inter-class separability in the embedding space.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.5,
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of classes
            s: Scale factor (default: 64.0)
            m: Angular margin (default: 0.5)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m

        # Class weight matrix (learnable)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m) for efficiency
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # Threshold for numerical stability
        self.mm = math.sin(math.pi - m) * m

    def set_margin(self, m: float):
        """Update margin value (for margin scheduling)"""
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            labels: Ground truth labels (B,). If None, returns cosine similarities only.

        Returns:
            If labels provided: Logits with angular margin applied (for training)
            If labels is None: Cosine similarities (for inference)
        """
        # Normalize weights
        normalized_weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: cos(theta) = embeddings @ weight.T
        cosine = F.linear(embeddings, normalized_weight)

        if labels is None:
            # Inference mode: return scaled cosine similarities
            return cosine * self.s

        # Training mode: apply angular margin
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability: when cos(theta) < cos(pi - m), use cos(theta) - mm
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin only to the correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale
        output = output * self.s

        return output


class EmbeddingModel(nn.Module):
    """
    Image classification model using EfficientNet V2 Large + ArcFace.

    Architecture:
    - Backbone: EfficientNet V2 Large (pretrained)
    - Embedding head: Linear layer (backbone_features -> embedding_dim)
    - L2 normalization on embeddings
    - ArcFace head for classification with angular margin
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
        arcface_s: float = 64.0,
        arcface_m: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of output classes
            embedding_dim: Dimension of embedding vector
            pretrained: Whether to use pretrained backbone
            arcface_s: ArcFace scale factor
            arcface_m: ArcFace angular margin
        """
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Load EfficientNet V2 Large backbone
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_v2_l(weights=weights)

        # Get backbone output features
        backbone_features = self.backbone.classifier[1].in_features

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # ArcFace head
        self.arcface = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            s=arcface_s,
            m=arcface_m,
        )

    def freeze_backbone(self):
        """Freeze backbone parameters for warm-up training"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized embeddings from input images.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            L2-normalized embeddings (B, embedding_dim)
        """
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
            labels: Ground truth labels (B,). If None, returns embeddings only.

        Returns:
            Tuple of (logits, embeddings)
            - logits: ArcFace output for loss computation (or cosine similarities if no labels)
            - embeddings: L2-normalized embedding vectors
        """
        embeddings = self.get_embeddings(x)
        logits = self.arcface(embeddings, labels)
        return logits, embeddings

    def set_arcface_margin(self, m: float):
        """Update ArcFace margin (for margin scheduling)"""
        self.arcface.set_margin(m)


def create_model(
    num_classes: int,
    embedding_dim: int = 512,
    pretrained: bool = True,
    arcface_s: float = 64.0,
    arcface_m: float = 0.5,
    device: str = "cuda"
) -> EmbeddingModel:
    """
    Create and initialize the model.

    Args:
        num_classes: Number of output classes
        embedding_dim: Dimension of embedding vector
        pretrained: Whether to use pretrained backbone
        arcface_s: ArcFace scale factor
        arcface_m: ArcFace angular margin
        device: Device to place model on

    Returns:
        Initialized model
    """
    model = EmbeddingModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        arcface_s=arcface_s,
        arcface_m=arcface_m,
    )
    return model.to(device)
