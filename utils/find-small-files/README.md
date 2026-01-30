# find_small_files

특정 크기 이하의 파일을 찾는 스크립트

## Usage

```bash
./run.sh <target_directory> <size> [-v] [-a]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `target_directory` | 검색할 디렉토리 경로 |
| `size` | 크기 기준 (예: 100k, 10M, 1G) |
| `-v` | (선택) 자세한 정보 출력 |
| `-a` | (선택) 숨김 파일 포함 (기본: 제외) |

## Examples

```bash
# 500KB 이하 파일 찾기
./run.sh ./images 500k

# 자세한 정보와 함께 출력
./run.sh ./images 10M -v

# 숨김 파일 포함
./run.sh ./images 500k -a

# 자세한 정보 + 숨김 파일 포함
./run.sh ./images 500k -v -a
```
