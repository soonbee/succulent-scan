import os
import shutil
import argparse

def main():
    # 1. 인자값 설정 (SOURCE 경로 입력)
    parser = argparse.ArgumentParser(description="Image data sampling script")
    parser.add_argument("source", help="Source directory path (e.g., ./result)")
    args = parser.parse_args()

    # --- 설정 구간 ---
    SOURCE_ROOT = args.source
    TARGET_DIR = "./sampled_data"
    PERCENT = 10
    # 비어있을 경우( [] ) SOURCE_ROOT 내의 모든 디렉터리를 자동으로 탐색함
    DIR_LIST = [] 
    # ----------------

    # 2. 대상 디렉터리 결정
    if not DIR_LIST:
        # SOURCE_ROOT 내의 항목 중 디렉터리만 필터링하여 리스트업
        DIR_LIST = [d for d in os.listdir(SOURCE_ROOT) 
                    if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
        print(f"💡 DIR_LIST가 비어있어 자동으로 {len(DIR_LIST)}개의 디렉터리를 찾았습니다.")

    # 3. 샘플링 작업 시작
    for folder in DIR_LIST:
        src_path = os.path.join(SOURCE_ROOT, folder)
        dst_path = os.path.join(TARGET_DIR, folder)

        # 대상 디렉터리 생성
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # 파일 목록 가져오기 및 정렬 (이름순)
        files = sorted([f for f in os.listdir(src_path) 
                        if os.path.isfile(os.path.join(src_path, f))])
        
        total_count = len(files)
        # 10% 계산 (최소 1개 보장)
        sample_count = max(1, int(total_count * PERCENT / 100)) if total_count > 0 else 0

        print(f"📂 [{folder}] 처리 중: 전체 {total_count}개 -> {PERCENT}%인 {sample_count}개 복사")

        # 상위 N개 파일 복사
        for i in range(sample_count):
            shutil.copy2(
                os.path.join(src_path, files[i]), 
                os.path.join(dst_path, files[i])
            )

    print("\n✅ 모든 작업이 완료되었습니다!")
    print(f"결과 저장 위치: {os.path.abspath(TARGET_DIR)}")

if __name__ == "__main__":
    main()