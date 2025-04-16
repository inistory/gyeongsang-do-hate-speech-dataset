import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, classification_report


model_path = "./detection_model_hard"
test_file = "gs_kold_test.json"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

all_preds = []
all_labels = []

for ex in test_data:
    text = ex["dialect"]
    span = ex["OFF_span"]

    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"][0]

    labels = [0] * len(input_ids[0])
    span_start = text.find(span)
    span_end = span_start + len(span) if span_start != -1 else -1

    for i, (start, end) in enumerate(offsets):
        if span_start != -1 and start >= span_start and end <= span_end:
            labels[i] = 1

    # 예측
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(logits, dim=-1)[0].tolist()

    for pred, label, (start, end) in zip(predictions, labels, offsets):
        if start == end:  # padding
            continue
        all_preds.append(pred)
        all_labels.append(label)

# ✅ 평가 결과 출력
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
print(f"\n📊 Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print("\n📄 Detailed Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["비욕설", "욕설"]))
