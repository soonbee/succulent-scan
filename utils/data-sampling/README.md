# Data Sampling Script

이 스크립트는 원본 디렉터리 내의 파일들을 이름순으로 정렬하여 상위 N%만큼 추출하여 복사합니다.

## Configuration
`main.py` 파일 상단의 설정을 수정하여 동작을 변경할 수 있습니다.
- `TARGET_DIR`: 추출된 파일이 저장될 경로 (기본: `./sampled_data`)
- `PERCENT`: 추출 비율 (기본: `10`)
- `DIR_LIST`: 리스트가 비어있을 경우 `SOURCE` 내 모든 디렉터리를 자동 탐색하며, 폴더명을 기입하면 해당 폴더만 처리합니다.

## Requirements
- Python 3.8+
- [uv](https://github.com/astral-sh/uv)

## Usage

### 1. 환경 동기화
```bash
uv sync
```

### 2. 스크립트 실행
원본 데이터가 위치한 경로를 인자로 전달하여 실행합니다.

# 기본 실행 구조
uv run main.py <SOURCE_PATH>

# 실행 예시
uv run main.py ./data
