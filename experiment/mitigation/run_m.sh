#!/bin/bash

# 모델 배열 정의
MODELS=(
    "beomi/KoAlpaca-Polyglot-5.8B"
    "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    "princeton-nlp/gemma-2-9b-it-SimPO"
    "Qwen/Qwen2.5-14B-Instruct"
)

# 결과 디렉토리 생성
mkdir -p ./results

# GPU 설정
export CUDA_VISIBLE_DEVICES=1

echo "================================================="
echo "            방언 완화 생성 시작           "
echo "================================================="

# 각 모델에 대해 반복
for MODEL in "${MODELS[@]}"; do
    # 모델 이름에서 경로 제외하고 마지막 부분만 추출
    MODEL_NAME=${MODEL##*/}
    
    # use_masked True로 실행
    echo "================================================="
    echo "            $MODEL_NAME 모델 실행 중 (with masking)          "
    echo "================================================="
    python generate_mitigated_dialect.py \
        --input_file "./masked_data.json" \
        --output_dir "./results" \
        --model_names "$MODEL" \
        --use_masked

    # use_masked False로 실행
    echo "================================================="
    echo "            $MODEL_NAME 모델 실행 중 (without masking)          "
    echo "================================================="
    python generate_mitigated_dialect.py \
        --input_file "./masked_data.json" \
        --output_dir "./results" \
        --model_names "$MODEL"
done

echo "================================================="
echo "            방언 완화 생성 완료           "
echo "=================================================" 