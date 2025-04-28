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
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using GPU: {gpu_id}")
    
    if model_name == "FacebookAI/roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
    elif model_name == "princeton-nlp/gemma-2-9b-it-SimPO":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    return model, tokenizer, device

def generate_mitigated_dialect(model, tokenizer, original_dialect, masked_dialect, offensive_span, model_name, device, use_masked=True, max_retries=3):
    for attempt in range(max_retries):
        # 프롬프트 구성
        if use_masked:
            prompt = f"""다음은 경상도 방언 문장입니다. 비속어를 완화하면서도 경상도 방언의 고유한 문체적 특성을 유지하는 문장을 생성해주세요.

표준어-방언 예시:
- 표준어: 그놈 -> 방언: 글마
- 표준어: 그놈들 -> 방언: 금마들
- 표준어: 그놈들이 -> 방언: 그놈아덜이
- 표준어: 그놈이 -> 방언: 금마가
- 표준어: 근데 -> 방언: 근디
- 표준어: 여기 -> 방언: 요기
- 표준어: 어디로 -> 방언: 어데로
- 표준어: 별로 -> 방언: 별루
- 표준어: 아주 -> 방언: 억시기
- 표준어: 정말 -> 방언: 진짜로
- 표준어: 뭐야 -> 방언: 머꼬
- 표준어: 알겠어 -> 방언: 알겠다아이가
- 표준어: 몰라 -> 방언: 모른다이가

원본 문장: {original_dialect}
원본 문장에서 비속어 표현이 마스킹된 문장: {masked_dialect}

주의사항:
1. 반드시 원본 문장보다 비속어 표현이 완화되어야 합니다.
2. 경상도 방언의 고유한 문체적 특성(어미, 조사, 발음 등)을 유지해야 합니다.
3. 문장의 의미는 원본과 동일하게 유지해야 합니다.
4. 원본 문장을 그대로 반복하지 말고, 반드시 완화된 표현으로 변경해야 합니다.
5. 비속어나 모욕적인 표현은 더 부드럽고 예의 바른 표현으로 대체해야 합니다.

완화된 문장:"""
        else:
            prompt = f"""다음은 경상도 방언 문장입니다. 비속어를 완화하면서도 경상도 방언의 고유한 문체적 특성을 유지하는 문장을 생성해주세요.

표준어-방언 예시:
- 표준어: 그놈 -> 방언: 글마
- 표준어: 그놈들 -> 방언: 금마들
- 표준어: 그놈들이 -> 방언: 그놈아덜이
- 표준어: 그놈이 -> 방언: 금마가
- 표준어: 근데 -> 방언: 근디
- 표준어: 여기 -> 방언: 요기
- 표준어: 어디로 -> 방언: 어데로
- 표준어: 별로 -> 방언: 별루
- 표준어: 아주 -> 방언: 억시기
- 표준어: 정말 -> 방언: 진짜로
- 표준어: 뭐야 -> 방언: 머꼬
- 표준어: 알겠어 -> 방언: 알겠다아이가
- 표준어: 몰라 -> 방언: 모른다이가

원본 문장: {original_dialect}

주의사항:
1. 반드시 원본 문장보다 비속어 표현이 완화되어야 합니다.
2. 경상도 방언의 고유한 문체적 특성(어미, 조사, 발음 등)을 유지해야 합니다.
3. 문장의 의미는 원본과 동일하게 유지해야 합니다.
4. 원본 문장을 그대로 반복하지 말고, 반드시 완화된 표현으로 변경해야 합니다.
5. 비속어나 모욕적인 표현은 더 부드럽고 예의 바른 표현으로 대체해야 합니다.

완화된 문장:"""

        # 입력 토큰화
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # GPU로 이동
        
        # 예측
        with torch.no_grad():
            if model_name == "FacebookAI/roberta-base":
                outputs = model(**inputs)
                predictions = outputs.logits
                predicted_tokens = torch.argmax(predictions, dim=-1)
                generated_text = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
            else:  # gemma 모델
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
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 완화된 문장만 추출
        try:
            generated_sentence = generated_text.split("완화된 문장:")[-1].strip()
            # 프롬프트가 포함된 경우 제거
            if "다음은 경상도 방언 문장입니다" in generated_sentence:
                generated_sentence = generated_sentence.split("다음은 경상도 방언 문장입니다")[0].strip()
            if "원본 문장:" in generated_sentence:
                generated_sentence = generated_sentence.split("원본 문장:")[0].strip()
            if "원본 문장에서 비속어 표현이 마스킹된 문장:" in generated_sentence:
                generated_sentence = generated_sentence.split("원본 문장에서 비속어 표현이 마스킹된 문장:")[0].strip()
            if "주의사항:" in generated_sentence:
                generated_sentence = generated_sentence.split("주의사항:")[0].strip()
        except:
            generated_sentence = generated_text.strip()
        
        # 생성된 문장이 비어있지 않고 원본 문장과 다른 경우에만 반환
        if generated_sentence and generated_sentence != original_dialect:
            return generated_sentence
        
        print(f"시도 {attempt + 1}/{max_retries}: 문장 생성 실패, 재시도 중...")
    
    # 모든 시도가 실패한 경우 원본 문장 반환
    print(f"최대 시도 횟수({max_retries})를 초과했습니다. 원본 문장을 반환합니다.")
    return original_dialect

def process_data(input_file, output_dir, model_names, use_masked):
    # 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 각 모델에 대해 실험
    for model_name in model_names:
        print(f"\n{model_name} 모델 로딩 중...")
        model, tokenizer, device = load_model_and_tokenizer(model_name)
        
        # 결과를 저장할 리스트
        results = []
        
        # 각 데이터에 대해 완화된 방언 생성
        for item in tqdm(data, desc=f"{model_name} {'마스킹 정보 있음' if use_masked else '마스킹 정보 없음'} 처리 중"):
            original_dialect = item['original_dialect']
            offensive_span = item['offensive_span']
            
            # 완화된 방언 생성
            mitigated_dialect = generate_mitigated_dialect(
                model, tokenizer, original_dialect, item.get('masked_dialect', ''), offensive_span, model_name, device, use_masked
            )
            
            # 결과 저장
            results.append({
                'original_dialect': original_dialect,
                'offensive_span': offensive_span,
                'mitigated_dialect': mitigated_dialect
            })
        
        # 모델 이름에서 파일 이름으로 사용할 부분 추출
        model_name_for_file = model_name.split('/')[-1]
        masked_str = "with_masked" if use_masked else "without_masked"
        output_file = f"{output_dir}/mitigated_data_{model_name_for_file}_{masked_str}.csv"
        
        # DataFrame 생성 및 CSV 저장
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n{model_name} {'마스킹 정보 있음' if use_masked else '마스킹 정보 없음'} 처리 완료")
        print(f"저장된 결과 수: {len(results)}")
        print(f"저장된 파일: {output_file}")

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