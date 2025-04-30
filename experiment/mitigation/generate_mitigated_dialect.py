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
            tokenizer.chat_template = "{{ messages[0]['content'] }}"
        
        device = "auto"  # device_map="auto"를 사용하는 경우
    
    return model, tokenizer, device

def generate_mitigated_dialect(model, tokenizer, original_dialect, masked_dialect, offensive_span, model_name, device, use_masked=True, max_retries=5):
    # OUTPUT 카운터 추가
    generate_mitigated_dialect.counter = getattr(generate_mitigated_dialect, 'counter', 0)
    
    # Construct prompt
    if use_masked:
        messages = [
            {
                "role": "user",
                "content": """This is a dialect mitigation task. Your task is to ONLY modify the parts marked with [MASK] while keeping ALL other parts EXACTLY the same.

IMPORTANT RULES:
1. ONLY modify the parts marked with [MASK]. DO NOT change any other parts of the text.
2. Keep the Gyeongsang-do dialect style consistent with the original text. Maintain the dialect's unique characteristics like '-노', '-나', '-다이', '-가이', etc.
3. Only mitigate offensive expressions within the [MASK] parts.
4. Maintain the original meaning of the sentence.
5. DO NOT modify any text that is not marked with [MASK].
6. Output ONLY in Korean. DO NOT include any English or other languages.
7. DO NOT include any explanations, rules, or additional text in your response.
8. DO NOT include [MASK] in your output. Replace [MASK] with the mitigated text.
9. Output ONLY the mitigated text. DO NOT include any explanations about why you made the changes or how you modified it.

Original text:
{original_dialect}

Masked text:
{masked_dialect}

[MITIGATED]:"""
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": """This is a dialect mitigation task. Your task is to ONLY modify the offensive parts while keeping ALL other parts EXACTLY the same.

IMPORTANT RULES:
1. ONLY modify the offensive expressions. DO NOT change any other parts of the text.
2. Keep the Gyeongsang-do dialect style consistent with the original text. Maintain the dialect's unique characteristics like '-노', '-나', '-다이', '-가이', etc.
3. Maintain the original meaning of the sentence.
4. DO NOT modify any text that is not offensive.
5. Output ONLY in Korean. DO NOT include any English or other languages.
6. DO NOT include any explanations, rules, or additional text in your response.
7. DO NOT include [MASK] in your output. Replace [MASK] with the mitigated text.
8. Output ONLY the mitigated text. DO NOT include any explanations about why you made the changes or how you modified it.

Original text:
{original_dialect}

[MITIGATED]:"""
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

    # 프롬프트 토큰 길이 계산
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_length = len(prompt_tokens)
    
    # max_length 설정 (프롬프트 토큰 길이 + 생성할 토큰 수)
    max_length = prompt_length + 100  # 100은 생성할 토큰 수
    
    # 모델별 생성 파라미터 설정
    if "qwen" in model_name.lower():
        generation_config = {
            "max_new_tokens": 100,  # 데이터셋의 최대 dialect 길이에 맞춤
            "max_length": max_length,  # 프롬프트 토큰 길이 + 생성 길이
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 2,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "repetition_penalty": 1.2
        }
    elif "gemma" in model_name.lower():
        generation_config = {
            "max_new_tokens": 100,  # 데이터셋의 최대 dialect 길이에 맞춤
            "max_length": max_length,  # 프롬프트 길이 + 생성 길이
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 2,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "repetition_penalty": 1.2
        }
    else:
        generation_config = {
            "max_new_tokens": 100,  # 데이터셋의 최대 dialect 길이에 맞춤
            "max_length": max_length,  # 프롬프트 길이 + 생성 길이
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 2,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "repetition_penalty": 1.2
        }

    # INPUT 토큰화 및 attention mask 설정
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    attention_mask = torch.ones_like(inputs["input_ids"])
    inputs["attention_mask"] = attention_mask
    
    for attempt in range(max_retries):
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
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # 디버깅: 생성된 텍스트 출력 (Examples 섹션 제외)
                    print("\n=== Generated Text ===")
                    print("Original:", original_dialect)
                    print("Masked:", masked_dialect if use_masked else "N/A")
                    # [MITIGATED]: 이후의 텍스트만 추출하고 special tokens 제거
                    mitigated_text = generated_text.split("[MITIGATED]:")[-1].strip()
                    mitigated_text = re.sub(r'<\|.*?\|>', '', mitigated_text)  # special tokens 제거
                    mitigated_text = re.sub(r'assistant', '', mitigated_text, flags=re.IGNORECASE)  # assistant 제거
                    mitigated_text = mitigated_text.strip()
                    print("Generated:", mitigated_text)
                    print("=====================\n")
                    
                    return mitigated_text
                
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
            mitigated_text = generate_mitigated_dialect(
                model, tokenizer, original_dialect, item.get('masked_dialect', ''), offensive_span, model_name, device, use_masked
            )
            
            # Add to results without parsing
            if mitigated_text:
                results.append({
                    'original_dialect': original_dialect,
                    'offensive_span': offensive_span,
                    'mitigated_dialect': mitigated_text.strip()
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