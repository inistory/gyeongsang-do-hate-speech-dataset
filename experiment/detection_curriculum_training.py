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
from typing import Optional
import json
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import torch.nn.functional as F

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="EleutherAI/polyglot-ko-5.8b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    base_model_name_or_path: str = field(
        default="EleutherAI/polyglot-ko-5.8b",
        metadata={"help": "Original base model name or path for config reference"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )

@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a json file)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "The input validation data file (a json file)."}
    )
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
        default="./output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    save_total_limit: Optional[int] = field(
        default=1,
        metadata={"help": "Limit the total amount of checkpoints."}
    )
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=500, metadata={"help": "Run an evaluation every X steps."})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when available"})
    dataloader_pin_memory: bool = field(default=False, metadata={"help": "Whether or not to pin memory for DataLoader"})

# 평가 함수
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels, true_preds = [], []
    for pred, label in zip(predictions, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                true_preds.append(p_)
                true_labels.append(l_)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='binary')
    acc = accuracy_score(true_labels, true_preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 전처리 함수
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

# 모델 정의
class TokenClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 모든 모듈을 같은 디바이스로 이동
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.classifier = self.classifier.to(device, dtype=torch.float16)

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
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.classifier.out_features)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def main():
    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # 첫 번째 GPU 사용
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 데이터셋 변수 초기화
    train_dataset = None
    valid_dataset = None
    test_dataset = None

    # 토크나이저 로드
    try:
        print(f"Loading tokenizer from: {model_args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            padding_side="right",
            use_fast=True  # Fast tokenizer 사용
        )
        print("Tokenizer loaded successfully")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        if tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        return

    # 데이터셋 로드 및 전처리
    def load_jsonl(file_path):
        if file_path is None:
            return None
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 파일 내용의 첫 문자를 확인하여 JSON 배열인지 JSONL인지 판단
                first_char = f.read(1)
                f.seek(0)  # 파일 포인터를 다시 처음으로
                
                if first_char == '[':  # JSON 배열 형식
                    data = json.load(f)
                else:  # JSONL 형식
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # 빈 줄 건너뛰기
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON at line {line_num}: {e}")
                            print(f"Line content: {line}")
                            continue
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        return data

    if training_args.do_train and data_args.train_file:
        train_data = load_jsonl(data_args.train_file)
        if train_data:
            train_dataset = Dataset.from_list(train_data)
            train_dataset = train_dataset.map(
                lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                batched=False
            )

    if training_args.do_eval and data_args.validation_file:
        valid_data = load_jsonl(data_args.validation_file)
        if valid_data:
            valid_dataset = Dataset.from_list(valid_data)
            valid_dataset = valid_dataset.map(
                lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                batched=False
            )

    if training_args.do_predict and data_args.test_file:
        test_data = load_jsonl(data_args.test_file)
        if test_data:
            test_dataset = Dataset.from_list(test_data)
            test_dataset = test_dataset.map(
                lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                batched=False
            )

    # 모델 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # base 모델 로드
    try:
        if training_args.do_train:
            print("Loading base model for training...")
            base_model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map={"": 0},  # 모든 모듈을 cuda:0에 할당
                trust_remote_code=model_args.trust_remote_code,
            )
        else:
            print("Loading model for evaluation/prediction...")
            base_model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.float16,
                device_map={"": 0},  # 모든 모듈을 cuda:0에 할당
                trust_remote_code=model_args.trust_remote_code
            )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # LoRA 설정
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query_key_value"],  # polyglot 모델의 attention 레이어
        lora_dropout=0.1,
        bias="none",
        task_type="TOKEN_CLS",
        inference_mode=False
    )
    lora_model = get_peft_model(base_model, lora_config)

    # 분류 모델 생성
    model = TokenClassificationModel(lora_model, hidden_size=lora_model.config.hidden_size, num_labels=2)

    # 트레이너 설정
    training_args.dataloader_pin_memory = False  # pin_memory 비활성화
    training_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        print(f"Validation file: {data_args.validation_file}")
        print(f"Output directory: {training_args.output_dir}")
        
        if valid_dataset is None:
            print("Error: No validation dataset available")
        else:
            print(f"Validation dataset size: {len(valid_dataset)}")
            try:
                # 출력 디렉토리 생성
                os.makedirs(training_args.output_dir, exist_ok=True)
                
                metrics = trainer.evaluate()
                print(f"Evaluation metrics: {metrics}")
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")

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
                # 출력 디렉토리 생성
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
