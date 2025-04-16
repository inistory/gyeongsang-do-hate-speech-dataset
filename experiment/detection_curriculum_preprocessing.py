import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# 1. 모델 및 토크나이저 설정
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 데이터 로드
with open("gs_kold.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 3. Token-level label 생성 함수
def get_token_labels(text, span):
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
    labels = [0] * len(tokens["input_ids"])
    if not span:
        return labels
    start_idx = text.find(span)
    if start_idx == -1:
        return labels
    end_idx = start_idx + len(span)
    for i, (start, end) in enumerate(tokens["offset_mapping"]):
        if start >= end_idx:
            break
        if start >= start_idx and end <= end_idx:
            labels[i] = 1
    return labels

# 4. 난이도 분류 기준
def get_difficulty(example):
    span_len = len(example["OFF_span"].split()) if example["OFF_span"] else 0
    text_len = len(example["dialect"].split())
    if span_len == 0:
        return "easy"
    if span_len == 1 and text_len <= 7:
        return "easy"
    elif span_len <= 2 and text_len <= 12:
        return "medium"
    else:
        return "hard"

# 5. 커리큘럼 분할
curriculum = {"easy": [], "medium": [], "hard": []}
for ex in raw_data:
    if ex["OFF"] and ex["OFF_span"]:
        labels = get_token_labels(ex["dialect"], ex["OFF_span"])
        level = get_difficulty(ex)
        curriculum[level].append({
            "text": ex["dialect"],
            "labels": labels
        })

# 6. DatasetDict 저장
datasets = DatasetDict({
    level: Dataset.from_list(samples)
    for level, samples in curriculum.items()
})

# 7. 저장
for level in ["easy", "medium", "hard"]:
    datasets[level].to_json(f"detection_{level}.json", orient="records", lines=True)

print("✅ 커리큘럼 기반 전처리 및 저장 완료")
