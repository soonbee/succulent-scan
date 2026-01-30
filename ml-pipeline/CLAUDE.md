# [Project] Image Retrieval-based Classification

## 1. Project Overview
* **Goal:** Develop an embedding model to classify images based on visual similarity.
* **Methodology:** EfficientNet V2 Backbone + ArcFace Metric Learning + Faiss Similarity Search.
* **Priority:** Accuracy is the top priority over inference speed or model size.
* **Hardware:** NVIDIA GPU with 24GB+ VRAM recommended.

## 2. Dataset Configuration
### Dataset Structure
Prepare your dataset in the following directory structure:
```
data/
├── class_a/
│   ├── id001_1.jpg
│   ├── id001_2.jpg
│   ├── id002_1.jpg
│   └── ...
├── class_b/
│   ├── id003_1.jpg
│   └── ...
└── class_c/
    └── ...
```

### Naming Convention
* **Directory Name:** Each subdirectory under `data/` represents a class label.
* **File Name Format:** `{ID}_{sequence}.jpg`
  * `ID`: Unique identifier for a group of related images (e.g., same object from different angles).
  * `sequence`: Numeric index to distinguish images within the same ID group.
* **Example:** `plant001_1.jpg`, `plant001_2.jpg` → Two images of the same plant (ID: plant001).

### Why ID Grouping Matters
Images with the same ID are treated as a **group** and will always be placed in the same split (train/val/test). This prevents **data leakage** where similar images of the same object appear in both training and validation sets.

### Data Pipeline & Splitting Rules
* **Directory Structure:** `data/{genus}/{ID}_{sequence}.jpg`
* **Split Ratio:** Train 70% / Validation 15% / Test 15%
* **Group Splitting:** Images with the same ID (prefix before '_') must stay together in Train, Validation, or Test set to prevent **Data Leakage**.
* **Stratified Split:** Maintain class distribution ratios across all splits.
* **Split Method:** Use `StratifiedGroupKFold` or custom group-aware stratified split. Priority: **Group Split > Stratified** (data leakage prevention is more critical).
* **Test Set:** Reserved for final service performance evaluation. Never used during training or model selection.
* **Split Storage:** Save split information as JSON files (run once, reuse for all experiments).
    ```
    splits/
    ├── train_ids.json    # List of IDs for training
    ├── val_ids.json      # List of IDs for validation
    └── test_ids.json     # List of IDs for final evaluation
    ```
* **Labeling:** Sort folder names alphabetically for indexing (0 to N-1) and save as `class_to_idx.json`.

### Image Preprocessing Policy
* **Target Size:** 480x480
* **Minimum Threshold:** 224x224 (EfficientNet minimum input size)
* **Exclusion Rule:** If any image in an ID group is below 224x224, **exclude the entire ID group** to maintain Group Split consistency.
* **Resize Method:** Bicubic interpolation for images between 224x224 and 480x480.
* **Data Augmentation (Train only):**
    * RandomResizedCrop (scale=0.8~1.0)
    * RandomHorizontalFlip (p=0.5)
    * RandomRotation (up to 15 degrees) - reduced from 30° for realistic camera tilt simulation
    * ColorJitter (brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05) - conservative to preserve plant color characteristics
    * Normalize (ImageNet mean/std)
    * **Note:** RandomVerticalFlip removed - unrealistic for side-view images with visible pots/gravity direction.
* **Validation Transform:** Resize (shorter side to 480, maintain aspect ratio) → CenterCrop(480) → Normalize.

## 3. Model Architecture & Training Strategy
### Model Specs
* **Backbone:** `EfficientNet V2 Large` (Pre-trained).
* **Embedding Head:** Remove Softmax; add a Linear layer with **512-dimensional output**.
* **Normalization:** Apply **L2 Normalization** to the embedding vector.
* **Loss Function:** `ArcFace Loss` with margin scheduling (s=64 fixed, m varies by phase).
    * **Note:** Default m=0.5, s=64 are standard values from face recognition. For small number of classes, consider lowering if training is unstable (e.g., m=0.3, s=48).

### Training Phase
1.  **Phase 1 (Warm-up, 5 Epochs):** Freeze Backbone, train ArcFace head with **m=0** (equivalent to Softmax CrossEntropy).
2.  **Phase 2 (Fine-tuning):** Unfreeze Backbone, train entire network with **ArcFace Loss (m=0.5)**.
* **Note:** Using single ArcFace layer with margin scheduling ensures weight continuity between phases.
* **Imbalance Handling:** Use `BalancedBatchSampler` (P×K format). P=number of classes, K=4~5 samples per class. Ensures class diversity within each batch, optimal for ArcFace.
* **Optimization:** Use `torch.cuda.amp` (Mixed Precision) and **Gradient Accumulation** (auto-configured based on GPU VRAM).
* **Batch Size:** P×K per step. With Gradient Accumulation step=4, effective batch size = batch_size × 4.
* **Learning Rate Scheduler:** `CosineAnnealingWarmRestarts` (T_0=10, T_mult=2) or `ReduceLROnPlateau` (patience=5, factor=0.5).
* **Early Stopping:** Monitor Recall@1 on validation set. Patience: 10 epochs without improvement.

### Checkpoint Strategy (Hybrid)
* **best_model.pt:** Save the model with the best validation metric (Recall@1).
* **latest_model.pt:** Save the most recent epoch for training resume.
* **checkpoint_epoch_N.pt:** Save every 10 epochs for ensemble candidates and overfitting analysis.
* **Contents:** Include `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, and `best_metric`.
* **Auto-Resume:** Training automatically resumes from `latest_model.pt` if it exists.
* **Fresh Start:** Use `--fresh` flag to ignore existing checkpoints and start new training.

### Cross-Platform Compatibility
* **Device Selection:** CUDA if available, otherwise CPU. MPS (Apple Silicon) is not supported due to Faiss compatibility issues.
* **Mixed Precision (AMP):** Enabled only on CUDA. Automatically disabled on CPU.
* **OpenMP Conflict Prevention:** Environment variables `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` are set in code to prevent PyTorch-Faiss OpenMP conflicts on non-CUDA systems.
* **pin_memory:** Enabled only on CUDA for efficient data transfer.

### GPU Auto-Detection
The training script automatically detects GPU capabilities and applies optimal settings:

| GPU | Precision | Workers | Grad Accum |
|-----|-----------|---------|------------|
| A100 80GB | BF16 | 16 | 1 |
| A100 40GB | BF16 | 8 | 2 |
| RTX 3090 24GB | FP16 | 4 | 4 |
| Other Ampere+ | BF16 | auto | auto |
| Pre-Ampere | FP16 | 4 | 4 |

* **BF16 (BFloat16):** Automatically enabled on Ampere+ GPUs (compute capability ≥ 8.0). Provides wider dynamic range than FP16, no GradScaler needed.
* **TF32:** Automatically enabled on Ampere+ GPUs for faster FP32 operations.
* **cudnn.benchmark:** Enabled for fixed input sizes (480×480) to optimize convolution algorithms.
* **Manual Override:** Use CLI arguments to override auto-detected settings:
    ```bash
    python train.py --use_bf16=false --num_workers=8 --accum_steps=2
    ```
* **Cross-GPU Compatibility:** Models trained on A100 (BF16) can be used on RTX 3090 (FP16) for inference without any modification. Weights are always saved in FP32.

## 4. Evaluation & Inference (Faiss)
### Evaluation Protocol
* **Gallery:** Train embeddings indexed in Faiss.
* **Query (Training phase):** Validation images → for model selection (Early Stopping, Best Model).
* **Query (Final evaluation):** Test images → for service performance measurement.
* **Recall@K:** Success if Top-K contains at least one correct class label. Averaged over all queries.

### Validation Metrics
* Primary: **Recall@1 and Recall@5** via Faiss search.
* Secondary: **mAP (mean Average Precision)** for ranking quality.
* Analysis: **Confusion Matrix** to identify frequently confused genus pairs.

### Faiss Configuration
* **Indexing:** Store trained embeddings in `Faiss IndexFlatL2`.
* **Search Function:** Implement a function that returns Top-K results for a query image.

---

## 5. [Reference] Post-processing Logic (For Future Implementation)
*The following logic should be considered during inference function design:*
1.  **Weighted Voting:** Determine final class using the sum of inverse distances ($1/d$) of the Top-20 results.
2.  **Reliability Threshold:** If Top-1 distance is too high or the margin between Top-1 and Top-2 is negligible, return 'Unclassifiable' or 'Reshoot Recommendation'.
