# Image Classification Pipeline

EfficientNet V2 + ArcFace 기반 이미지 분류 모델 학습 파이프라인

## Overview

이미지 검색 기반으로 이미지를 분류하는 임베딩 모델을 학습합니다.

- **Backbone:** EfficientNet V2 Large (pretrained)
- **Loss:** ArcFace with margin scheduling
- **Search:** Faiss IndexFlatL2

## Dataset Preparation

데이터를 다음과 같은 구조로 준비하세요:

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

### 파일 명명 규칙

- **디렉토리명:** `data/` 하위 각 폴더명이 클래스 라벨이 됩니다.
- **파일명 형식:** `{ID}_{sequence}.jpg`
  - `ID`: 동일 객체의 이미지 그룹을 식별하는 고유 ID
  - `sequence`: 같은 ID 내에서 이미지를 구분하는 번호
- **예시:** `plant001_1.jpg`, `plant001_2.jpg` → 같은 식물의 두 이미지 (ID: plant001)

### ID 그룹의 중요성

같은 ID를 가진 이미지들은 항상 동일한 split(train/val/test)에 배치됩니다. 이를 통해 동일 객체의 이미지가 학습과 검증 세트에 동시에 포함되는 **Data Leakage**를 방지합니다.

## Requirements

- Python 3.12+
- CUDA 지원 GPU (권장) 또는 CPU

## Installation

```bash
uv sync
```

## Usage

### 1. Split 생성 (최초 1회)

```bash
uv run python create_splits.py --data_dir ./data --output_dir ./splits
```

### 2. 학습

```bash
uv run python train.py --data_dir ./data --splits_dir ./splits
```

**체크포인트 Resume:**
- 학습이 중단되면 `latest_model.pt`에서 자동으로 이어서 학습
- 새로 시작하려면 `--fresh` 옵션 사용:
```bash
uv run python train.py --fresh
```

### 3. 평가

```bash
# Validation set
uv run python evaluate.py --checkpoint ./checkpoints/best_model.pt --split val

# Test set (최종 평가)
uv run python evaluate.py --checkpoint ./checkpoints/best_model.pt --split test
```

> **Note:** `uv run`은 가상환경을 자동으로 활성화합니다. 가상환경을 직접 활성화한 경우 `python` 명령어만 사용해도 됩니다.

## Project Structure

```
ml-pipeline/
├── config.py          # 하이퍼파라미터 설정
├── create_splits.py   # Train/Val/Test 분할
├── dataset.py         # Dataset, BalancedBatchSampler
├── model.py           # EfficientNet + ArcFace
├── train.py           # 학습 루프
├── evaluate.py        # Faiss 기반 평가
├── utils.py           # 유틸리티 함수
├── splits/            # Split 파일
└── checkpoints/       # 모델 체크포인트
```

## Environment Notes

### CUDA (권장)
- 최적 성능
- Mixed Precision (AMP) 자동 활성화
- 예상 학습 시간: 수 시간 (전체 데이터 기준)

### CPU
- 느리지만 정상 동작
- AMP 자동 비활성화
- OpenMP 충돌 자동 처리됨

### macOS / Windows
- CPU 모드로 동작
- MPS (Apple Silicon)는 Faiss 호환성 문제로 미지원

### GPU Auto-Detection

학습 스크립트가 GPU를 자동으로 감지하고 최적 설정을 적용합니다:

| GPU | Precision | Workers | Grad Accum |
|-----|-----------|---------|------------|
| A100 80GB | BF16 | 16 | 1 |
| A100 40GB | BF16 | 8 | 2 |
| RTX 3090 24GB | FP16 | 4 | 4 |
| 기타 Ampere+ | BF16 | auto | auto |
| Pre-Ampere | FP16 | 4 | 4 |

**자동 적용 최적화:**
- **BF16:** Ampere+ GPU (A100, RTX 30xx 등)에서 자동 활성화
- **TF32:** Ampere+ GPU에서 FP32 연산 가속
- **cudnn.benchmark:** 고정 입력 크기(480×480)에 최적화

**수동 오버라이드:**
```bash
# 자동 감지 사용 (권장)
uv run python train.py

# 수동 설정 (필요 시)
uv run python train.py --use_bf16=false --num_workers=8 --accum_steps=2
```

**Cross-GPU 호환성:**
- A100에서 학습한 모델을 RTX 3090에서 추론 가능
- 가중치는 항상 FP32로 저장됨

## Configuration

모든 설정은 `config.py`에서 변경 가능합니다.

### 데이터 설정 (DataConfig)

| 파라미터 | 기본값 | 설명 | 조정 기준 |
|---------|--------|------|----------|
| `image_size` | 480 | 입력 이미지 크기 | 224~600. 클수록 정확도↑, 메모리↑, 속도↓ |
| `min_image_size` | 224 | 최소 이미지 크기 | 이보다 작은 이미지가 있는 ID 그룹은 제외됨 |
| `train_ratio` | 0.70 | 학습 데이터 비율 | 데이터가 적으면 0.8까지 증가 가능 |
| `val_ratio` | 0.15 | 검증 데이터 비율 | train + val + test = 1.0 유지 |
| `test_ratio` | 0.15 | 테스트 데이터 비율 | 최종 성능 평가용, 학습에 사용 안 함 |
| `class_names` | None | 사용할 클래스 목록 | None이면 자동 탐색, 리스트 지정 시 해당 클래스만 사용 |

**class_names 사용 예시:**
```python
# 자동 탐색 (기본값) - data/ 하위 모든 디렉토리를 클래스로 인식
class_names: Optional[List[str]] = None

# 명시적 지정 - 지정한 클래스만 사용 (오타/누락 검증 가능)
class_names: Optional[List[str]] = field(default_factory=lambda: [
    "class_a",
    "class_b",
    "class_c",
])
```

### 모델 설정 (ModelConfig)

| 파라미터 | 기본값 | 설명 | 조정 기준 |
|---------|--------|------|----------|
| `embedding_dim` | 512 | 임베딩 벡터 차원 | 128~1024. 클래스가 많으면 증가 |
| `arcface_s` | 64.0 | ArcFace scale | 30~64. 학습 불안정 시 낮춤 |
| `arcface_m_warmup` | 0.0 | Warm-up 단계 margin | 0 고정 (Softmax와 동일) |
| `arcface_m_finetune` | 0.5 | Fine-tuning 단계 margin | 0.3~0.5. 클래스 수가 적거나 불안정 시 낮춤 |

**ArcFace 파라미터 참고:**
- `s` (scale): 클수록 결정 경계가 명확해지지만, 학습 초기에 불안정할 수 있음
- `m` (margin): 클수록 클래스 간 분리가 강해지지만, 수렴이 어려울 수 있음
- 클래스 수가 적으면 (10개 미만) `s=48`, `m=0.3` 정도로 낮추는 것을 권장

### 학습 설정 (TrainingConfig)

| 파라미터 | 기본값 | 설명 | 조정 기준 |
|---------|--------|------|----------|
| `warmup_epochs` | 5 | Backbone 고정 epoch 수 | 3~10. 데이터가 적으면 늘림 |
| `total_epochs` | 50 | 총 학습 epoch | Early stopping이 있으므로 넉넉하게 설정 |
| `num_classes_per_batch` | None | 배치 내 클래스 수 (P) | None이면 전체 클래스 사용. 클래스가 많으면 일부만 샘플링 |
| `samples_per_class` | 4 | 배치 내 클래스당 샘플 수 (K) | 2~8. GPU 메모리에 따라 조정 |
| `gradient_accumulation_steps` | None | Gradient 누적 횟수 | None이면 GPU 기준 자동 설정. 실제 배치 크기 = P×K×이 값 |
| `learning_rate` | 1e-4 | 학습률 | 1e-5~1e-3. 불안정 시 낮춤 |
| `weight_decay` | 1e-4 | L2 정규화 | 1e-5~1e-3. 과적합 시 높임 |
| `early_stopping_patience` | 10 | 성능 개선 없이 대기할 epoch | 5~20 |
| `use_bf16` | None | BFloat16 사용 여부 | None이면 GPU 기준 자동 설정 (Ampere+에서 활성화) |
| `num_workers` | None | DataLoader worker 수 | None이면 GPU 기준 자동 설정 |

**배치 크기 계산:**
```
batch_size = num_classes_per_batch (P) × samples_per_class (K)
effective_batch_size = batch_size × gradient_accumulation_steps
```
- `num_classes_per_batch=None`이면 P=전체 클래스 수
- `gradient_accumulation_steps=None`이면 GPU VRAM 기준 자동 설정
  - A100 80GB: 1, A100 40GB: 2, RTX 3090: 4
- 예시: 클래스 10개, K=4, accumulation=4 → effective batch size = 10×4×4 = 160
- GPU 메모리 부족 시 `samples_per_class` 또는 `gradient_accumulation_steps` 조정

### 스케줄러 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `scheduler_type` | `"cosine_warm_restarts"` | `"cosine_warm_restarts"` 또는 `"reduce_on_plateau"` |
| `cosine_t0` | 10 | Cosine annealing 주기 |
| `cosine_t_mult` | 2 | 주기 증가 배수 |
| `plateau_patience` | 5 | Plateau 감지 patience |
| `plateau_factor` | 0.5 | 학습률 감소 비율 |

### Augmentation 설정 (AugmentationConfig)

| 파라미터 | 기본값 | 설명 | 조정 기준 |
|---------|--------|------|----------|
| `crop_scale_min` | 0.8 | RandomResizedCrop 최소 비율 | 0.6~0.9. 작을수록 augmentation 강함 |
| `rotation_degrees` | 15 | 회전 각도 | 0~30. 실제 촬영 환경에 맞게 |
| `brightness` | 0.2 | 밝기 변화량 | 0~0.4 |
| `contrast` | 0.2 | 대비 변화량 | 0~0.4 |
| `saturation` | 0.1 | 채도 변화량 | 0~0.3. 색상이 중요한 경우 낮게 |
| `hue` | 0.05 | 색조 변화량 | 0~0.1. 색상이 중요한 경우 낮게 |

### 평가 설정 (EvaluationConfig)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `recall_k_values` | `[1, 5]` | 평가할 Recall@K 값 |
| `top_k` | 20 | 검색 시 반환할 최대 결과 수 |

## Troubleshooting

### "Illegal instruction" 에러 (OpenBLAS)

가상 머신이나 일부 오래된 CPU 환경에서 Faiss 실행 시 발생할 수 있습니다.

```bash
# 환경 변수 설정 후 실행
export OPENBLAS_CORETYPE=PRESCOTT
uv run python train.py --data_dir ./data --splits_dir ./splits
```

또는 스크립트 앞에 직접 추가:

```bash
OPENBLAS_CORETYPE=PRESCOTT uv run python train.py --data_dir ./data --splits_dir ./splits
```
