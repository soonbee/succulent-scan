# check-data-distribution

이미지 데이터셋의 분포를 분석하는 유틸리티입니다.

## 기능

- ID별 이미지 개수 통계 (min, max, mean, median)
- 클래스(폴더)별 이미지 수 집계
- 이미지 개수 분포 출력

## 사용법

```bash
uv run main.py <데이터셋_경로>
```

예시:
```bash
uv run main.py ./data
uv run main.py /path/to/dataset
```

## 파일명 형식

`{id}_{index}.{ext}` 형식의 이미지 파일을 인식합니다.

예: `1558591027_1.jpg`
