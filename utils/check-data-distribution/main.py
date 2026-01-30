#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter, defaultdict
import argparse
import re

# 파일명 예: 1558591027_1.jpg
FILENAME_RE = re.compile(r"^(?P<id>\d+)_(?P<idx>\d+)\.(?P<ext>jpe?g|png|webp)$", re.IGNORECASE)

def main() -> None:
    parser = argparse.ArgumentParser(description="이미지 데이터셋의 분포를 분석합니다.")
    parser.add_argument("target_dir", type=Path, help="분석할 데이터셋 디렉토리 경로")
    args = parser.parse_args()

    target_dir = args.target_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"target_dir not found: {target_dir}")

    per_id_count = Counter()
    per_class_count = defaultdict(int)  # 폴더명(클래스)별 이미지 수(참고용)
    total_files = 0
    matched_files = 0
    unmatched = []

    # ✅ data 하위 전체(폴더 깊이 상관없이) 스캔
    for p in target_dir.rglob("*"):
        if not p.is_file():
            continue
        total_files += 1

        m = FILENAME_RE.match(p.name)
        if not m:
            unmatched.append(str(p))
            continue

        matched_files += 1
        img_id = m.group("id")
        per_id_count[img_id] += 1

        # 클래스(최상위 폴더명) 집계: data/<class>/.../file
        try:
            rel = p.relative_to(target_dir)
            cls = rel.parts[0] if len(rel.parts) >= 2 else "(root)"
        except Exception:
            cls = "(unknown)"
        per_class_count[cls] += 1

    if not per_id_count:
        print(f"No matching image files under: {target_dir}")
        print(f"Total files scanned: {total_files}, matched: {matched_files}")
        if unmatched:
            print("\nExamples of unmatched (up to 20):")
            for n in unmatched[:20]:
                print(" -", n)
        return

    values = list(per_id_count.values())
    n_ids = len(values)
    total_imgs = sum(values)
    min_cnt = min(values)
    max_cnt = max(values)
    mean_cnt = total_imgs / n_ids

    values_sorted = sorted(values)
    mid = n_ids // 2
    median = values_sorted[mid] if n_ids % 2 == 1 else (values_sorted[mid - 1] + values_sorted[mid]) / 2

    dist = Counter(values)  # images_per_id -> num_ids

    print(f"Target dir: {target_dir.resolve()}")
    print(f"Total files scanned: {total_files}")
    print(f"Matched image files: {matched_files}")
    print(f"Unique IDs: {n_ids}")
    print(f"Total images (matched): {total_imgs}")
    print()

    print("Per-ID image count stats")
    print(f" - min   : {min_cnt}")
    print(f" - max   : {max_cnt}")
    print(f" - mean  : {mean_cnt:.4f}")
    print(f" - median: {median}")
    print()

    print("Distribution (images_per_id -> num_ids)")
    for k in sorted(dist.keys()):
        print(f" - {k:>3} -> {dist[k]}")
    print()

    print("Images per class folder (top-level under data/)")
    for cls, c in sorted(per_class_count.items(), key=lambda x: (-x[1], x[0])):
        print(f" - {cls}: {c}")

    print("\nSample IDs with smallest counts (up to 10)")
    for _id, c in sorted(per_id_count.items(), key=lambda x: (x[1], x[0]))[:10]:
        print(f" - {_id}: {c}")

    print("\nSample IDs with largest counts (up to 10)")
    for _id, c in sorted(per_id_count.items(), key=lambda x: (-x[1], x[0]))[:10]:
        print(f" - {_id}: {c}")

    if unmatched:
        print(f"\nUnmatched filenames/paths: {len(unmatched)} (showing up to 20)")
        for n in unmatched[:20]:
            print(" -", n)

if __name__ == "__main__":
    main()
