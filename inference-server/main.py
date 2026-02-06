import io
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Prevent OpenMP conflicts between PyTorch and Faiss
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Add ml-pipeline to path to reuse model architecture
ML_PIPELINE_DIR = str(Path(__file__).resolve().parent.parent / "ml-pipeline")
sys.path.insert(0, ML_PIPELINE_DIR)

from model import create_model

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".webp", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# OOD (Out-of-Distribution) detection thresholds
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", "1.0"))
MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "0.05"))
CONCENTRATION_THRESHOLD = float(os.environ.get("CONCENTRATION_THRESHOLD", "0.3"))

# Korean name mapping for the 7 succulent genera
KO_NAMES = {
    "aeonium": "에오니움",
    "dudleya": "두들레야",
    "echeveria": "에케베리아",
    "graptopetalum": "그랩토페탈룸",
    "haworthia": "하월시아",
    "lithops": "리톱스",
    "pachyphytum": "파키피튬",
}

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Env-configurable paths
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(Path(__file__).resolve().parent.parent / "ml-pipeline" / "checkpoints" / "best_model.pt"),
)
INDEX_DIR = os.environ.get(
    "INDEX_DIR",
    str(Path(__file__).resolve().parent.parent / "ml-pipeline" / "index"),
)

# Shared state populated during lifespan
state = {}


def build_transform(image_size: int = 480) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_dir = Path(INDEX_DIR)
    checkpoint_path = Path(CHECKPOINT_PATH)

    # Load class_to_idx
    class_to_idx_path = index_dir / "class_to_idx.json"
    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Load Faiss index
    index = faiss.read_index(str(index_dir / "gallery.index"))

    # Load gallery labels
    gallery_labels = np.load(str(index_dir / "gallery_labels.npy"))

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        embedding_dim = checkpoint_config.get("embedding_dim", 512)
    else:
        embedding_dim = 512

    model = create_model(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=False,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transform()

    state["model"] = model
    state["index"] = index
    state["gallery_labels"] = gallery_labels
    state["idx_to_class"] = idx_to_class
    state["transform"] = transform
    state["device"] = device

    yield

    state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/inference")
async def inference(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file extension '{ext}'."
                f" Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail="File size exceeds 10MB limit.",
        )

    # Load and preprocess image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = state["transform"](image).unsqueeze(0).to(state["device"])

    # Extract embedding and ArcFace logits
    with torch.no_grad():
        embedding = state["model"].get_embeddings(tensor)
        arcface_logits = state["model"].arcface(embedding)  # (1, num_classes)
    embedding_np = embedding.cpu().numpy().astype(np.float32)

    # Faiss Top-20 search
    distances, indices = state["index"].search(embedding_np, 20)
    distances = distances[0]
    indices = indices[0]

    # Weighted voting: sum of 1/(distance + epsilon) per class
    eps = 1e-6
    faiss_scores: dict[int, float] = {}
    for dist, idx in zip(distances, indices):
        label = int(state["gallery_labels"][idx])
        faiss_scores[label] = faiss_scores.get(label, 0.0) + 1.0 / (dist + eps)

    # OOD detection: check distance and margin reliability
    top1_distance = float(distances[0])
    distance_ok = top1_distance < DISTANCE_THRESHOLD

    sorted_scores = sorted(faiss_scores.values(), reverse=True)
    top1_score = sorted_scores[0]
    top2_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = (top1_score - top2_score) / (top1_score + 1e-9)
    margin_ok = margin > MARGIN_THRESHOLD

    # Top-K concentration: fraction of Top-20 neighbors belonging to the top-1 class
    top1_label = max(faiss_scores, key=faiss_scores.get)
    top1_count = sum(1 for idx in indices if int(state["gallery_labels"][idx]) == top1_label)
    concentration = top1_count / len(indices)
    concentration_ok = concentration > CONCENTRATION_THRESHOLD

    reliable = distance_ok and margin_ok and concentration_ok

    # ArcFace softmax probabilities for ranking absent classes
    arcface_probs = torch.softmax(arcface_logits, dim=1)[0].cpu().numpy()

    # Combine: Faiss scores for present classes, scaled ArcFace for absent
    min_faiss = min(faiss_scores.values()) if faiss_scores else 1.0
    class_scores: dict[int, float] = {}
    for label in state["idx_to_class"]:
        if label in faiss_scores:
            class_scores[label] = faiss_scores[label]
        else:
            class_scores[label] = float(arcface_probs[label]) * min_faiss * 0.01

    # Normalize to percentages
    total = sum(class_scores.values())
    results = []
    for label, score in class_scores.items():
        en_name = state["idx_to_class"][label]
        results.append({
            "ko": KO_NAMES.get(en_name, en_name),
            "en": en_name,
            "acc": round(score / total * 100),
        })

    # Sort by acc descending, take top 3
    results.sort(key=lambda x: x["acc"], reverse=True)
    results = results[:3]

    # Ensure acc >= 1 for all entries
    for r in results:
        if r["acc"] < 1:
            r["acc"] = 1

    # Ensure percentages sum to 100 by adjusting the top entry
    acc_sum = sum(r["acc"] for r in results)
    if acc_sum != 100:
        results[0]["acc"] += 100 - acc_sum

    return {"reliable": reliable, "results": results}
