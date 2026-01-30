#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

from PIL import Image

MIN_W = 480
MIN_H = 480
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def main() -> None:
    parser = argparse.ArgumentParser(description="이미지 해상도를 검사하여 최소 기준 미만인 파일을 찾습니다.")
    parser.add_argument("target_dir", type=Path, help="검사할 데이터셋 디렉토리 경로")
    args = parser.parse_args()

    target_dir = args.target_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"target_dir not found: {target_dir}")

    total_files = 0
    checked = 0
    failed = []
    smaller = []

    # ✅ data 하위 전체 스캔
    for p in target_dir.rglob("*"):
        if not p.is_file():
            continue
        total_files += 1

        if p.suffix.lower() not in EXTS:
            continue

        try:
            with Image.open(p) as img:
                w, h = img.size
            checked += 1

            if w < MIN_W or h < MIN_H:
                smaller.append((p, w, h))

        except Exception as e:
            failed.append((p, repr(e)))

    print(f"Target dir: {target_dir.resolve()}")
    print(f"Total files scanned: {total_files}")
    print(f"Images checked: {checked}")
    print(f"Failed to read: {len(failed)}")
    print(f"Smaller than {MIN_W}x{MIN_H}: {len(smaller)}")
    print()

    if smaller:
        print("List of smaller images (path -> WxH):")
        for p, w, h in sorted(smaller, key=lambda x: (x[1] * x[2], str(x[0]))):
            print(f" - {p} -> {w}x{h}")

    if failed:
        print("\nFailed files (up to 20):")
        for p, err in failed[:20]:
            print(f" - {p} -> {err}")

if __name__ == "__main__":
    main()
