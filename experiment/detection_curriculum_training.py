from transformers import AutoConfig  # 상단에 추가
import os  # 상단에 추가
from torch import nn
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments,
    DataCollatorForTokenClassification
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 모델 설정
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
label_all_tokens = False

# 평가 함수
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_preds = []
    for pred, label in zip(predictions, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:  # padding 무시
                true_preds.append(p_)
                true_labels.append(l_)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='binary')
    acc = accuracy_score(true_labels, true_preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 전처리 함수
def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    word_ids = tokenized.word_ids()
    labels = example["labels"]
    aligned_labels = []
    for idx in range(len(word_ids)):
        word_idx = word_ids[idx]
        if word_idx is None:
            aligned_labels.append(-100)
        elif label_all_tokens:
            aligned_labels.append(labels[word_idx])
        else:
            aligned_labels.append(labels[word_idx] if idx == 0 or word_ids[idx - 1] != word_idx else -100)
    tokenized["labels"] = aligned_labels
    return tokenized

# 단계별 학습
prev_model_path = None
for stage in ["easy", "medium", "hard"]:
    print(f"\n📚 Training on stage: {stage.upper()}")
    
    dataset = load_dataset("json", data_files={stage: f"detection_{stage}.json"})[stage]
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

    model = AutoModelForTokenClassification.from_pretrained(
        prev_model_path or model_name,
        num_labels=2
    )
    model.classifier = nn.Linear(model.config.hidden_size, 2)  # 출력 레이어를 2개 레이블로 재설정

    args = TrainingArguments(
        output_dir=f"./detection_model_{stage}",
        do_eval=True,                  # 평가 활성화
        do_train=True,                 # 학습 활성화
        save_steps=500,                # 저장 주기
        eval_steps=500,                # 평가 주기
        save_total_limit=1,            # 저장할 최대 체크포인트 수
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"./logs_{stage}",
        report_to="none"              # 추가 보고 비활성화
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(min(500, len(tokenized_dataset)))),
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # 체크포인트 경로 업데이트
    if trainer.state.best_model_checkpoint:
        prev_model_path = trainer.state.best_model_checkpoint
    else:
        # 최상의 체크포인트가 없는 경우 마지막 모델 사용
        trainer.save_model()  # 현재 모델 저장
        prev_model_path = args.output_dir
    
    print(f"✅ Saved best model to: {prev_model_path}")
