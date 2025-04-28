import os
import torch
import numpy as np
from torch import nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoModel, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForTokenClassification, BitsAndBytesConfig,
    HfArgumentParser, AutoConfig
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import argparse
from dataclasses import dataclass, field
from typing import Optional, List
import json
import torch.nn.functional as F

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="EleutherAI/polyglot-ko-5.8b")
    base_model_name_or_path: str = field(default="EleutherAI/polyglot-ko-5.8b")
    trust_remote_code: bool = field(default=True)

@dataclass
class DataArguments:
    train_file: Optional[str] = None
    train_files: Optional[List[str]] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    max_seq_length: int = 128
    curriculum_epochs: Optional[List[int]] = None

@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = "./output"
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = False
    save_total_limit: Optional[int] = 1
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    no_cuda: bool = False
    dataloader_pin_memory: bool = False
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    report_to: List[str] = field(default_factory=lambda: [])
    fp16: bool = True
    gradient_accumulation_steps: int = 8
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    run_name: str = ""


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    exact_matches = 0
    total_samples = len(predictions)
    for pred, label in zip(predictions, labels):
        if np.array_equal(pred[label != -100], label[label != -100]):
            exact_matches += 1
    return {
        "exact_match_ratio": exact_matches / total_samples,
        "exact_matches": exact_matches,
        "total_samples": total_samples
    }

def tokenize_and_align_labels(example, tokenizer, max_length=128, label_all_tokens=False):
    tokenized = tokenizer(example["dialect"], truncation=True, padding="max_length", max_length=max_length)
    word_ids = tokenized.word_ids()
    labels = example["labels"]
    aligned_labels = []
    for idx in range(len(word_ids)):
        word_idx = word_ids[idx]
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx < len(labels):
            if label_all_tokens:
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(labels[word_idx] if idx == 0 or word_ids[idx - 1] != word_idx else -100)
        else:
            aligned_labels.append(-100)
    tokenized["labels"] = aligned_labels
    return tokenized

class TokenClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 모든 모듈을 같은 디바이스로 이동
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model = self.base_model.to_empty(device=device)
        self.dropout = self.dropout.to_empty(device=device)
        self.classifier = self.classifier.to_empty(device=device)
        self.classifier = self.classifier.to(dtype=self.base_model.dtype)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 입력 텐서들을 현재 디바이스로 이동
        if input_ids is not None:
            input_ids = input_ids.to(self.base_model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.base_model.device)
        if labels is not None:
            labels = labels.to(self.base_model.device)

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 수치적 안정성을 위해 log_softmax 사용
            log_probs = F.log_softmax(logits, dim=-1)
            loss_fct = nn.NLLLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = log_probs.view(-1, self.classifier.out_features)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            # 손실 계산 전에 유효한 샘플이 있는지 확인
            if active_labels.numel() > 0:
                # 클래스 가중치 적용 (불균형 데이터셋 고려)
                class_weights = torch.tensor([1.0, 2.0], device=active_labels.device, dtype=active_logits.dtype)
                loss_fct = nn.NLLLoss(weight=class_weights)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 토크나이저 로드
    try:
        print(f"Loading tokenizer from: {model_args.model_name_or_path}")
        if model_args.model_name_or_path.startswith("./"):
            # 로컬 경로인 경우
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
                padding_side="right",
                use_fast=True,
                local_files_only=True,
                model_type="polyglot"
            )
        else:
            # Hugging Face 모델인 경우
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
                padding_side="right",
                use_fast=True,
                local_files_only=False,
                model_type="polyglot"
            )
        print("Tokenizer loaded successfully")
        
        if tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        return

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)

    # 데이터셋 로드 및 전처리
    def load_jsonl(file_path):
        if file_path is None:
            return None
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == '[':
                    data = json.load(f)
                else:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON at line {line_num}: {e}")
                            continue
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        return data

    # 학습 데이터셋 로드
    if training_args.do_train:
        if data_args.train_files:
            print("Loading curriculum training files...")
            train_data = []
            for file_path in data_args.train_files:
                file_data = load_jsonl(file_path)
                if file_data:
                    train_data.extend(file_data)
            if train_data:
                train_dataset = Dataset.from_list(train_data)
                train_dataset = train_dataset.map(
                    lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                    batched=False,
                    num_proc=1  # 병렬 처리 비활성화
                )
                print(f"Loaded {len(train_data)} training examples")
        elif data_args.train_file:
            print("Loading single training file...")
            train_data = load_jsonl(data_args.train_file)
            if train_data:
                train_dataset = Dataset.from_list(train_data)
                train_dataset = train_dataset.map(
                    lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                    batched=False,
                    num_proc=1  # 병렬 처리 비활성화
                )
                print(f"Loaded {len(train_data)} training examples")

    # 검증 데이터셋 로드
    if training_args.do_eval and data_args.validation_file:
        valid_data = load_jsonl(data_args.validation_file)
        if valid_data:
            valid_dataset = Dataset.from_list(valid_data)
            valid_dataset = valid_dataset.map(
                lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                batched=False,
                num_proc=1  # 병렬 처리 비활성화
            )
            print(f"Loaded {len(valid_data)} validation examples")

    # 테스트 데이터셋 로드
    if training_args.do_predict and data_args.test_file:
        test_data = load_jsonl(data_args.test_file)
        if test_data:
            test_dataset = Dataset.from_list(test_data)
            test_dataset = test_dataset.map(
                lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                batched=False,
                num_proc=1  # 병렬 처리 비활성화
            )
            print(f"Loaded {len(test_data)} test examples")

    base_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=model_args.trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True
    ).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.1,
        bias="none",
        task_type="TOKEN_CLS",
        inference_mode=False,
        init_lora_weights=True
    )
    lora_model = get_peft_model(base_model, lora_config)
    model = TokenClassificationModel(lora_model, hidden_size=lora_model.config.hidden_size, num_labels=2)

    # 트레이너 설정
    training_args.dataloader_pin_memory = False
    training_args.report_to = []  # wandb 로깅 비활성화
    training_args.fp16 = True
    training_args.gradient_accumulation_steps = 8
    training_args.per_device_train_batch_size = 2
    training_args.per_device_eval_batch_size = 2
    training_args.run_name = f"hybrid_{model_args.model_name_or_path}"  # run_name 설정

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 학습
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # 원본 모델의 config를 가져와서 수정
        original_config = AutoModel.from_pretrained(
            "EleutherAI/polyglot-ko-5.8b",
            trust_remote_code=True
        ).config
        
        # config 수정
        original_config.model_type = "polyglot"
        
        # config 저장
        original_config.save_pretrained(training_args.output_dir)
        
        # 모델과 토크나이저 저장
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # 평가
    if training_args.do_eval:
        print("\n=== Starting Evaluation ===")
        if valid_dataset is None:
            print("Error: No validation dataset available")
        else:
            metrics = trainer.evaluate()
            print(f"Evaluation metrics: {metrics}")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # 예측
    if training_args.do_predict:
        print("\n=== Starting Prediction ===")
        if test_dataset is None:
            print("Error: No test dataset available")
        else:
            predictions = trainer.predict(test_dataset=test_dataset)
            metrics = predictions.metrics
            print(f"Prediction metrics: {metrics}")
            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

if __name__ == "__main__":
    main()
