import os
import torch
import numpy as np
from torch import nn
from datasets import Dataset
from transformers import (
    AutoModel, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForTokenClassification, BitsAndBytesConfig,
    HfArgumentParser, AutoConfig
)
from dataclasses import dataclass, field
from typing import Optional
import json
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import torch.nn.functional as F

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="EleutherAI/polyglot-ko-5.8b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )

@dataclass
class DataArguments:
    test_file: Optional[str] = field(
        default=None, metadata={"help": "The input test data file (a json file)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./output_base",
        metadata={"help": "The output directory where the model predictions will be written."}
    )
    do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when available"})
    dataloader_pin_memory: bool = field(default=False, metadata={"help": "Whether or not to pin memory for DataLoader"})

# 평가 함수
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # 시퀀스 수준 평가
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

# 전처리 함수
def tokenize_and_align_labels(example, tokenizer, max_length=128):
    tokenized = tokenizer(example["dialect"], truncation=True, padding="max_length", max_length=max_length)
    word_ids = tokenized.word_ids()
    labels = example["labels"]
    aligned_labels = []

    for idx in range(len(word_ids)):
        word_idx = word_ids[idx]
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx < len(labels):
            aligned_labels.append(labels[word_idx] if idx == 0 or word_ids[idx - 1] != word_idx else -100)
        else:
            aligned_labels.append(-100)

    tokenized["labels"] = aligned_labels
    return tokenized

# 모델 정의
class TokenClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model = self.base_model.to(device)
        self.dropout = self.dropout.to(device)
        self.classifier = self.classifier.to(device)
        self.classifier = self.classifier.to(dtype=self.base_model.dtype)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
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
            log_probs = F.log_softmax(logits, dim=-1)
            active_loss = attention_mask.view(-1) == 1
            active_logits = log_probs.view(-1, self.classifier.out_features)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            if active_labels.numel() > 0:
                loss_fct = nn.NLLLoss()
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = torch.tensor(0.1, device=logits.device, dtype=logits.dtype, requires_grad=True)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def main():
    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 모델 이름에서 마지막 부분만 추출
    model_name = model_args.model_name_or_path.split('/')[-1]
    training_args.output_dir = f"./output_base_{model_name}_predict"

    # 토크나이저 로드
    try:
        print(f"Loading tokenizer from: {model_args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            padding_side="right",
            use_fast=True,
            local_files_only=True if model_args.model_name_or_path.startswith("./") else False,
            model_type="polyglot"
        )
        print("Tokenizer loaded successfully")
        
        if tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

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
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        return data

    # 테스트 데이터셋 로드
    test_dataset = None
    if data_args.test_file:
        test_data = load_jsonl(data_args.test_file)
        if test_data:
            test_dataset = Dataset.from_list(test_data)
            test_dataset = test_dataset.map(
                lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                batched=False
            )

    # 모델 설정
    try:
        print("Loading base model...")
        base_model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=model_args.trust_remote_code,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    model = TokenClassificationModel(base_model, hidden_size=base_model.config.hidden_size, num_labels=2)

    # 트레이너 설정
    training_args.dataloader_pin_memory = False
    training_args.report_to = []
    training_args.fp16 = False
    training_args.bf16 = True
    training_args.per_device_eval_batch_size = 1

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 예측
    if training_args.do_predict:
        print("\n=== Starting Prediction ===")
        print(f"Test file: {data_args.test_file}")
        print(f"Output directory: {training_args.output_dir}")
        
        if test_dataset is None:
            print("Error: No test dataset available")
        else:
            print(f"Test dataset size: {len(test_dataset)}")
            try:
                os.makedirs(training_args.output_dir, exist_ok=True)
                
                predictions = trainer.predict(test_dataset=test_dataset)
                metrics = predictions.metrics
                print(f"Prediction metrics: {metrics}")
                trainer.log_metrics("predict", metrics)
                trainer.save_metrics("predict", metrics)
            except Exception as e:
                print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main() 