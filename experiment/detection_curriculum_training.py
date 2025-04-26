import os
import torch
import numpy as np
from torch import nn
from datasets import Dataset
from transformers import (
    AutoModel, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForTokenClassification, BitsAndBytesConfig,
    HfArgumentParser
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dataclasses import dataclass, field
from typing import Optional, List
import json
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

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
    train_files: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "The input training data files for curriculum learning (json files)."}
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
    curriculum_epochs: Optional[List[int]] = field(
        default_factory=lambda: [1, 1, 1],
        metadata={"help": "Number of epochs to train on each difficulty level"}
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

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # 샘플별 정확도 계산을 위한 리스트
    sample_accuracies = []
    
    # 각 샘플별로 정확도 계산
    for pred, label in zip(predictions, labels):
        # 패딩이나 특수 토큰 제외
        valid_indices = [i for i, l in enumerate(label) if l != -100]
        if not valid_indices:
            continue
            
        # 해당 샘플의 유효한 토큰들만 선택
        valid_preds = [pred[i] for i in valid_indices]
        valid_labels = [label[i] for i in valid_indices]
        
        # 정확한 예측 수 계산
        correct_predictions = sum(1 for p, l in zip(valid_preds, valid_labels) if p == l)
        total_tokens = len(valid_indices)
        
        # 샘플별 정확도 계산
        sample_accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
        sample_accuracies.append(sample_accuracy)
    
    # 전체 샘플의 평균 정확도 계산
    mean_accuracy = np.mean(sample_accuracies) if sample_accuracies else 0
    
    # 기본 메트릭 계산 (전체 데이터셋 기준)
    token_predictions = []
    token_labels = []
    
    for pred, label in zip(predictions, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                token_predictions.append(p_)
                token_labels.append(l_)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        token_labels, 
        token_predictions, 
        average='binary'
    )
    acc = accuracy_score(token_labels, token_predictions)
    
    # 연속된 시퀀스 비교를 위한 변수 초기화
    total_sequences = 0
    correct_sequences = 0
    total_tokens = len(token_labels)
    
    # 연속된 시퀀스 비교
    i = 0
    while i < total_tokens:
        if token_labels[i] == 1:  # 혐오 표현 시퀀스 시작
            # 실제 시퀀스의 길이 계산
            actual_length = 0
            while i + actual_length < total_tokens and token_labels[i + actual_length] == 1:
                actual_length += 1
            
            # 예측된 시퀀스의 길이 계산
            pred_length = 0
            while i + pred_length < total_tokens and token_predictions[i + pred_length] == 1:
                pred_length += 1
            
            # 시퀀스가 정확히 일치하는지 확인
            if actual_length == pred_length:
                correct_sequences += 1
            total_sequences += 1
            
            i += actual_length
        else:
            i += 1
    
    # 시퀀스 정확도 계산
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
    
    # 혐오 표현 토큰 수 계산
    hate_tokens_predicted = sum(token_predictions)
    hate_tokens_actual = sum(token_labels)
    hate_tokens_correct = sum(1 for p, l in zip(token_predictions, token_labels) if p == l == 1)
    
    return {
        "accuracy": acc,
        "mean_sample_accuracy": mean_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sequence_accuracy": sequence_accuracy,
        "total_sequences": total_sequences,
        "correct_sequences": correct_sequences,
        "hate_tokens_predicted": hate_tokens_predicted,
        "hate_tokens_actual": hate_tokens_actual,
        "hate_tokens_correct": hate_tokens_correct
    }

def tokenize_and_align_labels(example, tokenizer, max_length=128, label_all_tokens=False):
    # 텍스트 토크나이징
    tokenized = tokenizer(
        example["dialect"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True
    )
    
    # 오프셋 매핑을 사용하여 토큰과 레이블 정렬
    offset_mapping = tokenized["offset_mapping"]
    labels = example["labels"]
    aligned_labels = []
    
    for offset in offset_mapping:
        if offset[0] == 0 and offset[1] == 0:  # [CLS] 토큰
            aligned_labels.append(-100)
        elif offset[0] == 0 and offset[1] == 0:  # [SEP] 토큰
            aligned_labels.append(-100)
        else:
            # 해당 토큰이 포함하는 문자 범위에 해당하는 레이블 찾기
            token_labels = []
            for i, (start, end) in enumerate(offset_mapping):
                if start < offset[1] and end > offset[0]:
                    if i < len(labels):
                        token_labels.append(labels[i])
            
            # 토큰에 해당하는 레이블이 있으면 1, 없으면 0
            aligned_labels.append(1 if any(l == 1 for l in token_labels) else 0)
    
    tokenized["labels"] = aligned_labels
    return tokenized

class TokenClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.classifier = self.classifier.to(device, dtype=torch.float16)

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
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.classifier.out_features)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    train_datasets = []
    valid_dataset = None
    test_dataset = None

    try:
        print(f"Loading tokenizer from: {model_args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/polyglot-ko-5.8b",
            trust_remote_code=True,
            padding_side="right",
            use_fast=True,
            local_files_only=False
        )
        print("Tokenizer loaded successfully")
        
        if tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

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

    if training_args.do_train and data_args.train_files:
        for train_file in data_args.train_files:
            curr_data = load_jsonl(train_file)
            if curr_data:
                curr_dataset = Dataset.from_list(curr_data)
                curr_dataset = curr_dataset.map(
                    lambda x: tokenize_and_align_labels(x, tokenizer, data_args.max_seq_length),
                    batched=False
                )
                train_datasets.append(curr_dataset)

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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        if training_args.do_train:
            print("Loading base model for training...")
            base_model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=model_args.trust_remote_code,
            )
        else:
            print("Loading model for evaluation/prediction...")
            base_model = AutoModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=model_args.trust_remote_code
            )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

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

    training_args.dataloader_pin_memory = False
    training_args.report_to = []

    if training_args.do_train:
        for difficulty_level, (curr_dataset, num_epochs) in enumerate(zip(train_datasets, data_args.curriculum_epochs)):
            print(f"\n=== Starting training on difficulty level {difficulty_level} for {num_epochs} epochs ===")
            
            training_args.num_train_epochs = num_epochs
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=curr_dataset,
                eval_dataset=valid_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=DataCollatorForTokenClassification(tokenizer),
                compute_metrics=compute_metrics,
            )
            
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.log_metrics(f"train_level_{difficulty_level}", metrics)
            trainer.save_metrics(f"train_level_{difficulty_level}", metrics)
            trainer.save_state()
            
            checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-level-{difficulty_level}")
            trainer.save_model(checkpoint_dir)
        
        original_config = AutoModel.from_pretrained(
            "EleutherAI/polyglot-ko-5.8b",
            trust_remote_code=True
        ).config
        original_config.model_type = "polyglot"
        original_config.save_pretrained(training_args.output_dir)
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    elif training_args.do_eval:
        print("\n=== Starting Evaluation ===")
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=compute_metrics,
        )
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        # 평가 모델 저장
        original_config = AutoModel.from_pretrained(
            "EleutherAI/polyglot-ko-5.8b",
            trust_remote_code=True
        ).config
        original_config.model_type = "polyglot"
        original_config.save_pretrained(training_args.output_dir)
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    elif training_args.do_predict:
        print("\n=== Starting Prediction ===")
        if test_dataset is None:
            print("Error: No test dataset available")
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                data_collator=DataCollatorForTokenClassification(tokenizer),
                compute_metrics=compute_metrics,
            )
            predictions = trainer.predict(test_dataset=test_dataset)
            metrics = predictions.metrics
            print(f"Prediction metrics: {metrics}")
            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)
            
            # 예측 모델 저장
            original_config = AutoModel.from_pretrained(
                "EleutherAI/polyglot-ko-5.8b",
                trust_remote_code=True
            ).config
            original_config.model_type = "polyglot"
            original_config.save_pretrained(training_args.output_dir)
            trainer.save_model(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
