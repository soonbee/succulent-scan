# check-image-resolution

이미지 해상도를 검사하여 최소 기준(480x480) 미만인 파일을 찾아내는 유틸리티입니다.

## 요구사항

- Python 3.12+
- Pillow

## 설치

```bash
uv sync
```

## 사용법

```bash
uv run main.py <데이터셋_경로>
```

예시:
```bash
uv run main.py ./data
uv run main.py /path/to/dataset
```

## 설정

`main.py`에서 다음 값을 수정하여 사용할 수 있습니다:

- `MIN_W`, `MIN_H`: 최소 해상도 기준 (기본값: 480x480)
- `EXTS`: 검사할 이미지 확장자
