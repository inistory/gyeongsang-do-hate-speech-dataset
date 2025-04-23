import json

# 분석할 파일 경로 리스트
file_paths = [
    "./detection_easy.json",
    "./detection_medium.json",
    "./detection_hard.json"
]

# 전체 카테고리별 개수를 저장할 딕셔너리 초기화
total_category_counts = {
    "untargeted": 0,
    "individual": 0,
    "group": 0,
    "other": 0
}

for file_path in file_paths:
    print(f"Processing file: {file_path}")
    category_counts = {
        "untargeted": 0,
        "individual": 0,
        "group": 0,
        "other": 0
    }

    # JSON 파일 읽기
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                tgt_value = data.get("TGT")
                # TGT 값이 유효한 카테고리인지 확인
                if tgt_value in {"untargeted", "individual", "group", "other"}:
                    category_counts[tgt_value] += 1
                    total_category_counts[tgt_value] += 1
                else:
                    print(f"Unexpected TGT value: {tgt_value} in file {file_path}")
            except json.JSONDecodeError:
                print("Invalid JSON line:", line)

    # 파일별 결과 출력
    print(f"Category counts for {file_path}:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    print()

# 전체 결과 출력
print("Total category counts across all files:")
for category, count in total_category_counts.items():
    print(f"{category}: {count}")