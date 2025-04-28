import json
from collections import defaultdict
import random

# 1. 데이터 로드'
with open("../korean_hatespeech/KOLD/data/gs_kold.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"전체 데이터 개수: {len(raw_data)}")


with open("gs_kold_train.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"gs_kold_train 개수: {len(raw_data)}")

def split_data_by_difficulty(raw_data):
    hate_examples = []
    non_hate_examples = []
    
    # hate와 non-hate 데이터 분리
    for ex in raw_data:
        item = {
            "standard": ex["standard"],
            "dialect": ex["dialect"],
            "OFF": ex["OFF"],
            "TGT": ex["TGT"],
            "OFF_span": ex["OFF_span"],
            "OFF_span_dialect": ex["OFF_span_dialect"],
            "labels": ex["labels"]  # 원본 데이터의 labels를 그대로 사용
        }
        
        if ex["OFF"] and ex["OFF_span"]:
            # labels에서 1의 개수를 세어 hate_count로 설정
            hate_word_count = sum(1 for label in item["labels"] if label == 1)
            item["hate_count"] = hate_word_count
            # OFF가 True이고 실제로 레이블이 있는 경우만 hate_examples에 추가
            if hate_word_count > 0:
                hate_examples.append(item)
            else:
                non_hate_examples.append(item)
        else:
            non_hate_examples.append(item)
                
    # 전체 데이터를 3등분
    total_per_level = len(raw_data) // 3
    
    # hate 데이터 정렬
    hate_examples.sort(key=lambda x: x["hate_count"])
    hate_chunk_size = len(hate_examples) // 3
    
    # hate 데이터 3등분
    hate_splits = {
        "easy": hate_examples[:hate_chunk_size],
        "medium": hate_examples[hate_chunk_size:hate_chunk_size*2],
        "hard": hate_examples[hate_chunk_size*2:]
    }
    
    # non-hate 데이터 랜덤 셔플
    random.shuffle(non_hate_examples)
    
    # 각 난이도별로 필요한 non-hate 데이터 수 계산
    non_hate_splits = {}
    for level in ["easy", "medium", "hard"]:
        needed_non_hate = total_per_level - len(hate_splits[level])
        if level == "easy":
            non_hate_splits[level] = non_hate_examples[:needed_non_hate]
            current_idx = needed_non_hate
        elif level == "medium":
            non_hate_splits[level] = non_hate_examples[current_idx:current_idx + needed_non_hate]
            current_idx += needed_non_hate
        else:  # hard
            non_hate_splits[level] = non_hate_examples[current_idx:]
    
    # 최종 결과 병합
    curriculum = defaultdict(list)
    for level in ["easy", "medium", "hard"]:
        curriculum[level].extend(hate_splits[level])
        curriculum[level].extend(non_hate_splits[level])
        # 각 난이도별 데이터 셔플
        random.shuffle(curriculum[level])
    
    return curriculum

# 5. 커리큘럼 분할 수정
curriculum = split_data_by_difficulty(raw_data)

# full 버전 추가
curriculum["full"] = raw_data

# 6. 저장
for level in ["easy", "medium", "hard", "full"]:
    # JSON Lines 형식으로 저장 (UTF-8 인코딩 명시)
    with open(f"detection_{level}.jsonl", "w", encoding="utf-8") as f:
        for item in curriculum[level]:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    # 각 난이도별 전체 데이터 수
    total_count = len(curriculum[level])
    # hate와 non-hate 데이터 수 계산
    hate_count = sum(1 for item in curriculum[level] if 1 in item['labels'])
    non_hate_count = total_count - hate_count
    
    # hate 데이터의 통계 계산
    hate_items = [item for item in curriculum[level] if 1 in item['labels']]
    hate_counts = [sum(1 for label in item['labels'] if label == 1) for item in hate_items]
    min_hate = min(hate_counts) if hate_counts else 0
    avg_hate = sum(hate_counts) / len(hate_counts) if hate_counts else 0
    max_hate = max(hate_counts) if hate_counts else 0
    
    print(f"✓ {level} 난이도:")
    print(f"  - 전체 데이터 수: {total_count}")
    print(f"  - Hate 데이터 수: {hate_count}")
    print(f"  - Non-hate 데이터 수: {non_hate_count}")
    print(f"  - Hate 개수 통계:")
    print(f"    * 최소: {min_hate}")
    print(f"    * 평균: {avg_hate:.2f}")
    print(f"    * 최대: {max_hate}")
    print()

print("✅ 커리큘럼 기반 전처리 및 저장 완료")