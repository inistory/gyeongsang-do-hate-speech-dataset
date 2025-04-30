import json
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForCausalLM
import re
import pandas as pd
from tqdm import tqdm
import os
import argparse

def load_model_and_tokenizer(model_name):
    # CUDA_VISIBLE_DEVICES 환경 변수에서 GPU 번호 가져오기
    gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    print(f"Using GPU: {gpu_id}")
    
    if model_name == "FacebookAI/roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForMaskedLM.from_pretrained(model_name).cuda()
        device = torch.device(f"cuda")
    else:
        # 모델별 최적화 설정
        if "polyglot" in model_name.lower():
            # KoAlpaca-Polyglot 모델 설정
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=True  # 8비트 양자화로 메모리 사용량 감소
            )
        elif "llama" in model_name.lower():
            # Llama 모델 설정
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=True
            )
        elif "gemma" in model_name.lower():
            # Gemma 모델 설정
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        elif "qwen" in model_name.lower():
            # Qwen 모델 설정
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=True
            )
        else:
            # 기본 설정
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        # 채팅 템플릿 설정
        if "polyglot" in model_name.lower() or "gemma" in model_name.lower():
            tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}### Human: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}### Assistant: {{ message['content'] }}

{% endif %}{% endfor %}"""
        
        device = "auto"  # device_map="auto"를 사용하는 경우
    
    return model, tokenizer, device

def generate_mitigated_dialect(model, tokenizer, original_dialect, masked_dialect, offensive_span, model_name, device, use_masked=True, max_retries=5):
    # OUTPUT 카운터 추가
    generate_mitigated_dialect.counter = getattr(generate_mitigated_dialect, 'counter', 0)
    
    for attempt in range(max_retries):
        # Construct prompt
        if use_masked:
            messages = [
                {
                    "role": "user",
                    "content": """This is a dialect mitigation task. Please only mitigate the masked parts while keeping the rest of the text unchanged.

Rules:
1. Only mitigate the masked parts ([MASK]) and keep the rest of the text unchanged
2. Preserve the dialect as much as possible
3. Only mitigate offensive expressions
4. Maintain the original meaning of the sentence

Examples:
Original text: 저놈 머리통을 확 깨뿔라카노
Masked text: 저[MASK]을 확 깨뿔라카노
Mitigated text: [MITIGATED]: 저분 머리를 확 쥐어박고 싶다카이

Original text: 개같은 새끼
Masked text: [MASK]
Mitigated text: [MITIGATED]: 못된 사람

Original text: 저놈이가 미쳤나
Masked text: 저[MASK]
Mitigated text: [MITIGATED]: 저분이가 정신이 없네

Original text: 씨발놈아
Masked text: [MASK]
Mitigated text: [MITIGATED]: 아이고야

Original text: 죽여뿔라카다
Masked text: [MASK]뿔라카다
Mitigated text: [MITIGATED]: 혼내뿔라카다

Original text:
{original_dialect}

Masked text:
{masked_dialect}

Please write the mitigated version after [MITIGATED]:, only changing the masked parts."""
                },
                {
                    "role": "assistant",
                    "content": "[MITIGATED]:"
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": """This is a dialect mitigation task. Please only mitigate the offensive parts while keeping the rest of the text unchanged.

Rules:
1. Only mitigate offensive expressions
2. Preserve the dialect as much as possible
3. Maintain the original meaning of the sentence
4. Keep the rest of the text unchanged

Examples:
Original text: 저놈 머리통을 확 깨뿔라카노
Mitigated text: [MITIGATED]: 저분 머리를 확 쥐어박고 싶다카이

Original text: 개같은 새끼
Mitigated text: [MITIGATED]: 못된 사람

Original text: 저놈이가 미쳤나
Mitigated text: [MITIGATED]: 저분이가 정신이 없네

Original text: 씨발놈아
Mitigated text: [MITIGATED]: 아이고야

Original text: 죽여뿔라카다
Mitigated text: [MITIGATED]: 혼내뿔라카다

Original text:
{original_dialect}

Please write the mitigated version after [MITIGATED]:, only changing the offensive parts."""
                },
                {
                    "role": "assistant",
                    "content": "[MITIGATED]:"
                }
            ]

        # Format the messages with the actual values
        messages[0]["content"] = messages[0]["content"].format(
            original_dialect=original_dialect,
            masked_dialect=masked_dialect if use_masked else ""
        )

        # 채팅 템플릿 적용
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # INPUT 토큰화 및 attention mask 설정
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
        attention_mask = torch.ones_like(inputs["input_ids"])
        inputs["attention_mask"] = attention_mask
        
        # 예측
        with torch.no_grad():
            if model_name == "FacebookAI/roberta-base":
                # RoBERTa 모델용 디바이스 처리
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                predictions = outputs.logits
                predicted_tokens = torch.argmax(predictions, dim=-1)
                generated_text = tokenizer.decode(predicted_tokens[0])
            else:  # Qwen이나 Gemma 모델
                # device_map="auto"를 사용하는 모델용 디바이스 처리
                if hasattr(model, 'device'):
                    model_device = model.device
                elif hasattr(model, 'module.device'):
                    model_device = model.module.device
                else:
                    # 모델의 첫 번째 파라미터의 디바이스 사용
                    for param in model.parameters():
                        model_device = param.device
                        break
                
                # INPUT을 모델의 디바이스로 이동
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                try:
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_config,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
                    
                    # 디버깅: 생성된 텍스트 출력
                    print("\n=== Generated Text ===")
                    print("Original:", original_dialect)
                    print("Masked:", masked_dialect if use_masked else "N/A")
                    print("Generated:", generated_text)
                    print("=====================\n")
                    
                    return generated_text
                
                except Exception as e:
                    print(f"Generation error: {str(e)}")
                    continue
        
        print(f"Attempt {attempt + 1}/{max_retries}: Failed to generate sentence, retrying...")
    
    # Return None if all attempts fail
    print(f"Exceeded maximum attempts ({max_retries}). Failed to generate mitigated sentence.")
    return None

def process_data(input_file, output_dir, model_names, use_masked):
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each model
    for model_name in model_names:
        print(f"\nLoading model {model_name}...")
        model, tokenizer, device = load_model_and_tokenizer(model_name)
        
        # Initialize results list
        results = []
        
        # Process each item
        for item in tqdm(data, desc=f"Processing {model_name} {'with' if use_masked else 'without'} masking"):
            original_dialect = item['original_dialect']
            offensive_span = item['offensive_span']
            
            # Generate mitigated dialect
            mitigated_dialect = generate_mitigated_dialect(
                model, tokenizer, original_dialect, item.get('masked_dialect', ''), offensive_span, model_name, device, use_masked
            )
            
            # Add to results without parsing
            if mitigated_dialect:
                results.append({
                    'original_dialect': original_dialect,
                    'offensive_span': offensive_span,
                    'mitigated_dialect': mitigated_dialect
                })
            else:
                print(f"Failed to generate mitigated version for: {original_dialect}")
        
        if results:
            # Create output filename
            model_name_for_file = model_name.split('/')[-1]
            masked_str = "with_masked" if use_masked else "without_masked"
            output_file = f"{output_dir}/mitigated_data_{model_name_for_file}_{masked_str}.csv"
            
            # Save to CSV
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"\nProcessing complete for {model_name}")
            print(f"Successfully processed sentences: {len(results)}")
            print(f"Results saved to: {output_file}")
        else:
            print(f"\nNo valid results generated for {model_name}")

if __name__ == "__main__":
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description='방언 완화 생성 스크립트')
    parser.add_argument('--input_file', type=str, default='./masked_data.json',
                        help='INPUT 데이터 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--model_names', type=str, nargs='+',
                        default=["princeton-nlp/gemma-2-9b-it-SimPO", "FacebookAI/roberta-base"],
                        help='사용할 모델 이름 리스트')
    parser.add_argument('--use_masked', action='store_true',
                        help='마스킹 정보 사용 여부')
    
    args = parser.parse_args()
    
    # 결과 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    process_data(args.input_file, args.output_dir, args.model_names, args.use_masked) 