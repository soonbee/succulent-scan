# [Project] Succulent Genus Classification Inference Server

## 1. Overview
* FastAPI-based inference server. Accepts succulent plant images and returns genus classification results.
* Loads TorchScript model and Faiss index from `ml-pipeline`.
* Single-file architecture (`main.py`).

## 2. Running
```bash
cd inference-server
uv run uvicorn main:app --host 0.0.0.0 --port 6000
```

## 3. API

### `GET /healthz`
Health check. Returns `{"status": "ok", "timestamp": "..."}`.

### `POST /inference`
* **Content-Type:** `multipart/form-data`
* **Field:** `file` (image file)
* **Allowed extensions:** `.jpg`, `.jpeg`, `.webp`, `.png`
* **Max size:** 10MB

#### Response Format
```json
{
  "reliable": true,
  "results": [
    {"ko": "에케베리아", "en": "echeveria", "acc": 92},
    {"ko": "에오니움", "en": "aeonium", "acc": 6},
    {"ko": "하월시아", "en": "haworthia", "acc": 2}
  ]
}
```
* `reliable`: OOD detection result. `false` indicates the image is likely not a succulent.
* `results`: Top 3 classes. `acc` values always sum to 100.

## 4. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEX_DIR` | `../ml-pipeline/index` | Index directory (`model.pt`, `gallery.index`, `gallery_labels.npy`, `class_to_idx.json`) |
| `DISTANCE_THRESHOLD` | `1.0` | OOD detection: upper bound for Top-1 Faiss distance |
| `MARGIN_THRESHOLD` | `0.05` | OOD detection: lower bound for weighted vote score margin between rank 1 and 2 |
| `CONCENTRATION_THRESHOLD` | `0.3` | OOD detection: lower bound for Top-K neighbor concentration of the top-1 class |

## 5. Inference Pipeline
1. Preprocessing: Resize(480) -> CenterCrop(480) -> Normalize(ImageNet)
2. EfficientNet V2 embedding extraction (512-dim, L2-normalized)
3. Faiss Top-20 search -> weighted voting (1/distance)
4. OOD detection: distance threshold + margin threshold + Top-K concentration to determine `reliable`
5. Classes absent from Faiss results are supplemented via ArcFace softmax
6. Top 3 classes normalized to percentages and returned

## 6. Dependencies
* Python >= 3.12
* Model loaded via `torch.jit.load` (TorchScript) — no dependency on `ml-pipeline` source code
* Package management: `uv` (`pyproject.toml` + `uv.lock`)

## 7. Code Style
* Formatter: `black` (line-length=88)
* All logic lives in a single file (`main.py`). Do not split until complexity warrants it.
