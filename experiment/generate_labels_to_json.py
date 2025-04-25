import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# 1. 모델 및 토크나이저 설정
model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Token-level label 생성 함수
def get_token_labels(text, span):
    if not span:
        tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
        return [0] * len(tokenized["input_ids"])
    
    text = text.strip().lower()
    spans = [s.strip().lower() for s in span.split(",")]
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
    labels = [0] * len(tokens["input_ids"])
    
    for sub_span in spans:
        start_idx = text.find(sub_span)
        if start_idx == -1:
            print(f"[에러] 스팬 매칭 실패: Text: {text}, Sub-span: {sub_span}")
            raise ValueError(f"텍스트에서 스팬을 찾을 수 없습니다.")
        
        end_idx = start_idx + len(sub_span)
        for idx, (start, end) in enumerate(tokens["offset_mapping"]):
            if start == 0 and end == 0:
                continue
            if start < end_idx and end > start_idx:
                labels[idx] = 1
    
    return labels

# 3. 데이터 로드
with open("/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/gs_kold.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 4. 욕설 + span이 있는 샘플만 필터링
filtered = [ex for ex in raw_data if ex["OFF"] and ex["OFF_span"]]

# 5. 학습용(train), 검증용(valid), 평가용(test) 분할
train_data, temp_data = train_test_split(filtered, test_size=0.2, random_state=42)  # 80% train, 20% temp
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 10% valid, 10% test

# 6. labels 필드 추가
for dataset in [train_data, valid_data, test_data]:
    for item in dataset:
        item["labels"] = get_token_labels(item["dialect"], item["OFF_span_dialect"])

# 7. JSON 저장
with open("gs_kold_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("gs_kold_valid.json", "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)

with open("gs_kold_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("✅ 데이터 분할 및 labels 추가 완료: train, valid, test JSON 저장")