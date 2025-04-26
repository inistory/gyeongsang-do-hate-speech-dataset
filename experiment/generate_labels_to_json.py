import json
from sklearn.model_selection import train_test_split

# 1. Token-level label 생성 함수
def get_token_labels(text, span):
    if not span:
        return [0] * len(text)
    
    text = text.strip()
    span = span.strip()
    labels = [0] * len(text)
    
    # span의 시작 위치 찾기
    start_idx = text.find(span)
    if start_idx == -1:
        print(f"[에러] 스팬 매칭 실패: Text: {text}, Span: {span}")
        raise ValueError(f"텍스트에서 스팬을 찾을 수 없습니다.")
    
    # span 길이만큼 1로 레이블링
    for i in range(start_idx, start_idx + len(span)):
        if i < len(labels):
            labels[i] = 1
    
    return labels

# 2. 데이터 로드
with open("/home/jungin/workspace/gyeongsang-do-hate-speech-dataset/korean_hatespeech/KOLD/data/gs_kold.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 3. 욕설 + span이 있는 샘플만 필터링
filtered = [ex for ex in raw_data if ex["OFF"] and ex["OFF_span"]]

# 4. 학습용(train), 검증용(valid), 평가용(test) 분할
train_data, temp_data = train_test_split(filtered, test_size=0.2, random_state=42)  # 80% train, 20% temp
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 10% valid, 10% test

# 5. labels 필드 추가
for dataset in [train_data, valid_data, test_data]:
    for item in dataset:
        item["labels"] = get_token_labels(item["dialect"], item["OFF_span_dialect"])

# 6. JSON 저장
with open("gs_kold_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("gs_kold_valid.json", "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)

with open("gs_kold_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("✅ 데이터 분할 및 labels 추가 완료: train, valid, test JSON 저장")