import os
import torch
import numpy as np
from torch import nn
from datasets import Dataset
from transformers import (
    AutoModel, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForTokenClassification, BitsAndBytesConfig,
    HfArgumentParser, EarlyStoppingCallback
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
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set."}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation on the validation set."}
    )
    output_dir: str = field(
        default="./output_curriculum",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for AdamW if we apply some."}
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
class CustomTrainingArguments(TrainingArguments):
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set."}
    )
    metric_for_best_model: str = field(
        default="mean_sample_accuracy",
        metadata={"help": "The metric to use to compare models."}
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether to load the best model found during training at the end of training."}
    )
    greater_is_better: bool = field(
        default=True,
        metadata={"help": "Whether a larger value of the metric is better."}
    )
    dataloader_pin_memory: bool = field(
        default=False,
        metadata={"help": "Whether or not to pin memory for DataLoader"}
    )
    report_to: List[str] = field(
        default_factory=list,
        metadata={"help": "The list of integrations to report the results and logs to."}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of update steps between two evaluations."}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Number of updates steps before two checkpoint saves."}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."}
    )

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
        
        # 문자 단위로 정확도 계산
        correct_chars = 0
        total_chars = len(valid_labels)
        
        for p, l in zip(valid_preds, valid_labels):
            if p == l:
                correct_chars += 1
        
        # 샘플별 정확도 계산
        sample_accuracy = correct_chars / total_chars if total_chars > 0 else 0
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
        max_length=max_length
    )
    
    # 원본 데이터의 labels 사용
    original_labels = example["labels"]
    
    # 토큰화된 텍스트의 길이에 맞게 레이블 조정
    aligned_labels = []
    for i in range(max_length):
        if i < len(original_labels):
            aligned_labels.append(original_labels[i])
        else:
            aligned_labels.append(-100)  # 패딩 토큰은 -100으로 레이블링
    
    tokenized["labels"] = aligned_labels
    return tokenized

class TokenClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 가중치 초기화
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
        
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
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.classifier.out_features)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            # 손실 계산 전에 유효한 레이블이 있는지 확인
            if active_labels.numel() > 0:
                loss = loss_fct(active_logits, active_labels)
            else:
                # 유효한 레이블이 없는 경우 기본 손실값 설정
                loss = torch.tensor(0.0, device=active_logits.device, requires_grad=True)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # TrainingArguments 직접 생성
    training_args = CustomTrainingArguments(
        output_dir=model_args.output_dir,
        do_train=model_args.do_train,
        do_predict=model_args.do_predict,
        save_total_limit=1,
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
        no_cuda=False,
        dataloader_pin_memory=False,
        metric_for_best_model="mean_sample_accuracy",
        load_best_model_at_end=False,
        remove_unused_columns=True,
        label_names=["labels"],
        report_to=[],
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        seed=42,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_accumulation_steps=1,
        save_safetensors=True,
        save_only_model=True
    )

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
                            item = json.loads(line)
                            # standard 필드 제외하고 필요한 필드만 사용
                            processed_item = {
                                "dialect": item["dialect"],
                                "OFF_span_dialect": item.get("OFF_span_dialect", ""),
                                "labels": item.get("labels", [0] * len(item["dialect"]))
                            }
                            data.append(processed_item)
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

    if training_args.do_train and data_args.validation_file:
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
            print("Loading model for prediction...")
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

    if training_args.do_train:
        for difficulty_level, (curr_dataset, num_epochs) in enumerate(zip(train_datasets, data_args.curriculum_epochs)):
            print(f"\n=== Starting training on difficulty level {difficulty_level} for {num_epochs} epochs ===")
            
            training_args.num_train_epochs = num_epochs
            
            # 데이터셋 크기에 따라 eval_steps와 save_steps 계산
            num_update_steps_per_epoch = len(curr_dataset) // training_args.per_device_train_batch_size
            training_args.eval_steps = num_update_steps_per_epoch  # 매 epoch마다 평가
            training_args.save_steps = num_update_steps_per_epoch  # 매 epoch마다 저장
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=curr_dataset,
                eval_dataset=valid_dataset,
                tokenizer=tokenizer,
                data_collator=DataCollatorForTokenClassification(tokenizer),
                compute_metrics=compute_metrics
            )
            
            # 학습 시작
            best_metric = float('-inf')
            best_checkpoint = None
            
            train_result = trainer.train()
            metrics = train_result.metrics
            
            # 현재 모델의 성능 평가
            eval_metrics = trainer.evaluate()
            current_metric = eval_metrics.get("eval_mean_sample_accuracy", 0)
            
            # 최고 성능 갱신 여부 확인
            if current_metric > best_metric:
                best_metric = current_metric
                best_checkpoint = os.path.join(training_args.output_dir, f"checkpoint-level-{difficulty_level}")
                trainer.save_model(best_checkpoint)
                print(f"New best model saved at {best_checkpoint} with metric: {best_metric}")
            
            trainer.log_metrics(f"train_level_{difficulty_level}", metrics)
            trainer.save_metrics(f"train_level_{difficulty_level}", metrics)
            trainer.save_state()
            
            # 최고 성능 모델 로드
            if best_checkpoint is not None:
                print(f"Loading best model from {best_checkpoint} with metric: {best_metric}")
                base_model = AutoModel.from_pretrained(
                    best_checkpoint,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=model_args.trust_remote_code
                )
                lora_model = get_peft_model(base_model, lora_config)
                model = TokenClassificationModel(lora_model, hidden_size=lora_model.config.hidden_size, num_labels=2)
            
            # 최종 모델 저장
            final_checkpoint_dir = os.path.join(training_args.output_dir, f"final-level-{difficulty_level}")
            trainer.save_model(final_checkpoint_dir)
            print(f"Final model saved at {final_checkpoint_dir}")
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
