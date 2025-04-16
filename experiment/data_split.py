import json
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터 로드
with open("gs_kold.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. 욕설 + span이 있는 샘플만 필터링
filtered = [ex for ex in raw_data if ex["OFF"] and ex["OFF_span"]]

# 3. 학습용(train), 평가용(test) 분할
train_data, test_data = train_test_split(filtered, test_size=0.1, random_state=42)

# 4. JSON 저장
with open("gs_kold_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("gs_kold_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# 5. CSV 파일로도 저장 (human-readable)
df_test = pd.DataFrame({
    "text": [ex["dialect"] for ex in test_data],
    "span": [ex["OFF_span"] for ex in test_data]
})
df_test.to_csv("gs_kold_test.csv", index=False, encoding="utf-8-sig")

print("✅ 데이터 분할 및 저장 완료: train, test JSON + test CSV")
