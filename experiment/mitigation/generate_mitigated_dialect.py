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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        device = "auto"  # device_map="auto"를 사용하는 경우
    # else:
    #     raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    return model, tokenizer, device

def generate_mitigated_dialect(model, tokenizer, original_dialect, masked_dialect, offensive_span, model_name, device, use_masked=True, max_retries=3):
    # 출력 카운터 추가
    generate_mitigated_dialect.counter = getattr(generate_mitigated_dialect, 'counter', 0)
    
    for attempt in range(max_retries):
        # Construct prompt
        if use_masked:
            messages = [
                {
                    "role": "user",
                    "content": """Here are examples of converting offensive Gyeongsang dialect sentences. Always respond in Korean:

저놈 머리통을 확 깨뿔라카노 -> [MITIGATED]: 저분한테 한마디 해뿔라카이
개같은 새끼 -> [MITIGATED]: 못된 사람아
저놈이가 미쳤나 -> [MITIGATED]: 저분이가 정신이 없네
씨발놈아 -> [MITIGATED]: 아이고야
죽여뿔라카다 -> [MITIGATED]: 혼내뿔라카다

Original sentence:
{original_dialect}

Masked sentence (offensive words are masked with [MASK]):
{masked_dialect}

Use the masked sentence to identify offensive parts and generate a less offensive version in Korean.
Only provide the mitigated sentence using the format:
[MITIGATED]: (mitigated sentence)"""
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
                    "content": """Here are examples of converting offensive Gyeongsang dialect sentences. Always respond in Korean:

저놈 머리통을 확 깨뿔라카노 -> [MITIGATED]: 저분한테 한마디 해뿔라카이
개같은 새끼 -> [MITIGATED]: 못된 사람아
저놈이가 미쳤나 -> [MITIGATED]: 저분이가 정신이 없네
씨발놈아 -> [MITIGATED]: 아이고야
죽여뿔라카다 -> [MITIGATED]: 혼내뿔라카다

Convert this sentence (respond in Korean only):
{original_dialect}

Only provide the mitigated sentence in Korean using the format:
[MITIGATED]: (mitigated sentence)"""
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

        # 입력 토큰화
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
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
                
                # 입력을 모델의 디바이스로 이동
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(outputs[0])
                
                # 처음 3번만 출력
                if generate_mitigated_dialect.counter < 3:
                    print(f"\n생성된 원본 텍스트:\n{generated_text}\n")
                    generate_mitigated_dialect.counter += 1
        
        # Extract mitigated sentence
        try:
            if model_name == "Qwen/Qwen2.5-14B-Instruct":
                # Qwen 모델의 출력에서 응답 추출 개선
                parts = generated_text.split("assistant")
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
                else:
                    print("Qwen 모델 응답에서 'assistant' 구분자를 찾을 수 없습니다.")
                    continue
            
            # [MITIGATED]: 형식 검색 개선
            match = re.search(r'\[MITIGATED\]:\s*([^\n]+)', generated_text, re.IGNORECASE)
            if not match:
                # 전체 텍스트에서 의미 있는 응답 찾기 시도
                lines = generated_text.split('\n')
                for line in lines:
                    if len(line.strip()) > 10 and re.search('[가-힣]', line):
                        generated_text = line.strip()
                        break
                else:
                    print(f"유효한 응답을 찾을 수 없습니다. 전체 출력:\n{generated_text}")
                    continue
            else:
                generated_text = match.group(1).strip()
            
            # Remove special characters and quotes
            generated_text = re.sub(r'["\']', '', generated_text)
            
            # Keep only Korean text and basic punctuation
            generated_text = re.sub(r'[^가-힣\s\.,\?!]', '', generated_text)
            
            # Remove leading/trailing whitespace and normalize spaces
            generated_text = ' '.join(generated_text.split())
            
            # Validation checks
            if not generated_text or len(generated_text) < 5:
                print(f"Generated text too short or empty. Retrying...")
                continue
            
            if generated_text == original_dialect:
                print(f"Generated text identical to original. Retrying...")
                continue
            
            if not re.search('[가-힣]', generated_text):
                print(f"No Korean text found in output. Retrying...")
                continue
            
            return generated_text
            
        except Exception as e:
            print(f"Error during parsing: {str(e)}")
            continue
        
        print(f"Attempt {attempt + 1}/{max_retries}: Failed to generate sentence, retrying...")
    
    # Return original sentence if all attempts fail
    print(f"Exceeded maximum attempts ({max_retries}). Returning original sentence.")
    return original_dialect

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
            
            # Only add to results if mitigated text is different from original and valid
            if mitigated_dialect and mitigated_dialect != original_dialect:
                # [MITIGATED]: 이후 부분만 추출
                if '[MITIGATED]:' in mitigated_dialect:
                    mitigated_dialect = mitigated_dialect.split('[MITIGATED]:')[1].strip()
                
                results.append({
                    'original_dialect': original_dialect,
                    'offensive_span': offensive_span,
                    'mitigated_dialect': mitigated_dialect
                })
        
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
                        help='입력 데이터 파일 경로')
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