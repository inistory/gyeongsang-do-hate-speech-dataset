import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from collections import defaultdict
import random

# 1. 모델 및 토크나이저 설정
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 데이터 로드
with open("gs_kold_train.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 3. Token-level label 생성 함수
def get_token_labels(text, span):
    # OFF_span_dialect이 없는 경우만 0으로 레이블링
    if not span:
        tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
        return [0] * len(tokenized["input_ids"])
    
    # 전처리: 공백 제거 및 소문자 변환
    text = text.strip().lower()
    spans = [s.strip().lower() for s in span.split(",")]  # ","로 분리하여 각각 처리
    
    # 토크나이징 수행
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
    labels = [0] * len(tokens["input_ids"])
    
    for sub_span in spans:
        # 텍스트에서 각 sub_span의 시작과 끝 위치 찾기
        start_idx = text.find(sub_span)
        if start_idx == -1:
            # 디버깅 로그 추가 
            print(f"[에러] 스팬 매칭 실패:")
            print(f"  Text: {text}")
            print(f"  Sub-span: {sub_span}")
            raise ValueError(f"텍스트에서 스팬을 찾을 수 없습니다.")
        
        end_idx = start_idx + len(sub_span)
        
        # offset_mapping을 사용하여 각 토큰이 sub_span에 포함되는지 확인
        for idx, (start, end) in enumerate(tokens["offset_mapping"]):
            # [CLS], [SEP] 등의 특수 토큰은 (0,0)으로 매핑됨
            if start == 0 and end == 0:
                continue
            # 토큰이 sub_span과 겹치는지 확인
            if end <= start_idx:  # sub_span 이전의 토큰
                continue
            if start >= end_idx:  # sub_span 이후의 토큰
                break
            if start < end_idx and end > start_idx:  # sub_span과 겹치는 토큰
                labels[idx] = 1
    
    return labels


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
            "labels": get_token_labels(ex["dialect"], ex["OFF_span_dialect"])
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

# 6. DatasetDict 저장
datasets = DatasetDict({
    level: Dataset.from_list([{
        **item,  # 기존 item의 모든 필드를 포함
        "labels": item["labels"]  # labels 추가
    } for item in samples])
    for level, samples in curriculum.items()
})

# 7. 저장
for level in ["easy", "medium", "hard"]:
    # JSON Lines 형식으로 저장 (UTF-8 인코딩 명시)
    with open(f"detection_{level}.json", "w", encoding="utf-8") as f:
        for item in datasets[level]:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    # 각 난이도별 전체 데이터 수
    total_count = len(datasets[level])
    # hate와 non-hate 데이터 수 계산
    hate_count = sum(1 for item in datasets[level] if 1 in item['labels'])
    non_hate_count = total_count - hate_count
    
    print(f"✓ {level} 난이도:")
    print(f"  - 전체 데이터 수: {total_count}")
    print(f"  - Hate 데이터 수: {hate_count}")
    print(f"  - Non-hate 데이터 수: {non_hate_count}")
    print()

print("✅ 커리큘럼 기반 전처리 및 저장 완료")