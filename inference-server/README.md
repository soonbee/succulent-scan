# Succulent Genus Classification Inference Server

다육식물 이미지를 받아 속(genus) 분류 결과를 반환하는 FastAPI 기반 추론 서버.

EfficientNet V2 임베딩 + Faiss 검색 + OOD 탐지 파이프라인으로 7개 속을 분류한다.

## 지원 속(Genus)

| 한국어       | English       |
| ------------ | ------------- |
| 에케베리아   | Echeveria     |
| 에오니움     | Aeonium       |
| 하월시아     | Haworthia     |
| 리톱스       | Lithops       |
| 파키피튬     | Pachyphytum   |
| 그랩토페탈룸 | Graptopetalum |
| 두들레야     | Dudleya       |

## 요구사항

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- 인덱스 파일: `model.pt`, `gallery.index`, `gallery_labels.npy`, `class_to_idx.json`

## 로컬 실행

```bash
cd inference-server
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 6000
```

```bash
# INDEX_DIR 환경변수가 필수. 인덱스 파일 경로를 지정한다.
INDEX_DIR=./index uv run uvicorn main:app --host 0.0.0.0 --port 6000
```

## Docker

### 빌드

`DEVICE` build arg로 CPU/GPU 이미지를 선택할 수 있다. 기본값은 `gpu`.

```bash
# GPU 빌드 (기본)
docker build -t inference-server:gpu .

# CPU 빌드 (이미지 ~2GB 작음)
docker build --build-arg DEVICE=cpu -t inference-server:cpu .
```

### 실행

인덱스 디렉토리를 `/data/index`에 마운트해야 한다.

```bash
# CPU
docker run -p 6000:6000 \
  -v ./index:/data/index \
  inference-server:cpu

# GPU (NVIDIA Container Toolkit 필요)
docker run --gpus all -p 6000:6000 \
  -v ./index:/data/index \
  inference-server:gpu
```

## API

### `GET /healthz`

헬스체크 엔드포인트.

```bash
curl http://localhost:6000/healthz
```

```json
{ "status": "ok", "timestamp": "2025-01-01T00:00:00+00:00" }
```

### `POST /inference`

이미지 파일을 받아 분류 결과를 반환한다.

- **Content-Type:** `multipart/form-data`
- **Field:** `file` (이미지 파일)
- **허용 확장자:** `.jpg`, `.jpeg`, `.webp`, `.png`
- **최대 크기:** 10MB

```bash
curl -X POST http://localhost:6000/inference \
  -F "file=@succulent.jpg"
```

```json
{
  "reliable": true,
  "results": [
    { "ko": "에케베리아", "en": "echeveria", "acc": 92 },
    { "ko": "에오니움", "en": "aeonium", "acc": 6 },
    { "ko": "하월시아", "en": "haworthia", "acc": 2 }
  ]
}
```

- `reliable`: OOD 탐지 결과. `false`이면 다육식물이 아닌 이미지일 가능성이 높음.
- `results`: 상위 3개 클래스. `acc` 합은 항상 100.

## 환경변수

| 변수                      | 기본값     | 설명                                              |
| ------------------------- | ---------- | ------------------------------------------------- |
| `INDEX_DIR`               | **(필수)** | 인덱스 디렉토리 경로 (Docker에서는 `/data/index`) |
| `DISTANCE_THRESHOLD`      | `1.0`      | OOD 탐지: Top-1 Faiss 거리 상한                   |
| `MARGIN_THRESHOLD`        | `0.05`     | OOD 탐지: 1위-2위 가중 투표 점수 마진 하한        |
| `CONCENTRATION_THRESHOLD` | `0.3`      | OOD 탐지: Top-K 이웃 중 1위 클래스 집중도 하한    |

## 추론 파이프라인

1. 전처리: Resize(480) → CenterCrop(480) → Normalize(ImageNet)
2. EfficientNet V2로 512차원 임베딩 추출 (L2 정규화)
3. Faiss Top-20 검색 → 가중 투표 (1/distance)
4. OOD 탐지: 거리 임계값 + 마진 임계값 + Top-K 집중도로 `reliable` 판정
5. Faiss 결과에 없는 클래스는 ArcFace softmax로 보완
6. 상위 3개 클래스를 백분율로 정규화하여 반환
