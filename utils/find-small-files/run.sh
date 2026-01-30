#!/bin/bash

# 특정 크기 이하의 파일을 찾는 스크립트
# Usage: ./find_small_files.sh <target_directory> <size> [-v] [-a]
# Example: ./find_small_files.sh ./images 500k -v -a

usage() {
    echo "Usage: $0 <target_directory> <size> [-v] [-a]"
    echo ""
    echo "Arguments:"
    echo "  target_directory  검색할 디렉토리 경로"
    echo "  size              크기 기준 (예: 100k, 10M, 1G)"
    echo "  -v                (선택) 자세한 정보 출력"
    echo "  -a                (선택) 숨김 파일 포함 (기본: 제외)"
    echo ""
    echo "Example:"
    echo "  $0 ./images 500k"
    echo "  $0 ./images 10M -v"
    echo "  $0 ./images 500k -a      # 숨김 파일 포함"
    echo "  $0 ./images 500k -v -a   # 자세한 정보 + 숨김 파일 포함"
    exit 1
}

# 인자 개수 확인
if [ $# -lt 2 ]; then
    usage
fi

TARGET_DIR="$1"
SIZE="$2"
VERBOSE=false
INCLUDE_HIDDEN=false

# 옵션 확인 (-v, -a)
shift 2
while [ $# -gt 0 ]; do
    case "$1" in
        -v) VERBOSE=true ;;
        -a) INCLUDE_HIDDEN=true ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# 디렉토리 존재 확인
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' 디렉토리가 존재하지 않습니다."
    exit 1
fi

echo "=== ${SIZE} 이하 파일 검색: ${TARGET_DIR} ==="
if [ "$INCLUDE_HIDDEN" = true ]; then
    echo "(숨김 파일 포함)"
else
    echo "(숨김 파일 제외)"
fi
echo ""

# 숨김 파일 제외 조건 설정
if [ "$INCLUDE_HIDDEN" = true ]; then
    HIDDEN_FILTER=""
else
    HIDDEN_FILTER="-not -path '*/.*'"
fi

if [ "$VERBOSE" = true ]; then
    # 자세한 정보 출력 (-exec 활용)
    if [ "$INCLUDE_HIDDEN" = true ]; then
        find "$TARGET_DIR" -type f -size -"$SIZE" -exec ls -lh {} +
    else
        find "$TARGET_DIR" -type f -size -"$SIZE" -not -path '*/.*' -exec ls -lh {} +
    fi
else
    # 파일 경로만 출력
    if [ "$INCLUDE_HIDDEN" = true ]; then
        find "$TARGET_DIR" -type f -size -"$SIZE"
    else
        find "$TARGET_DIR" -type f -size -"$SIZE" -not -path '*/.*'
    fi
fi
