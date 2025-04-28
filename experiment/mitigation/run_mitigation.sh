#!/bin/bash

# 결과 디렉토리 생성
mkdir -p ./results

# GPU 0번 사용 설정
export CUDA_VISIBLE_DEVICES=0

echo "================================================="
echo "            방언 완화 생성 시작           "
echo "================================================="
echo "사용 GPU: 0번"
echo "================================================="

# Gemma 모델로 실행 (마스킹 정보 있음)
echo "================================================="
echo "            Gemma 모델 실행 중 (마스킹 정보 있음)           "
echo "================================================="
python generate_mitigated_dialect.py \
    --input_file "./masked_data.json" \
    --output_dir "./results" \
    --model_names "princeton-nlp/gemma-2-9b-it-SimPO" \
    --use_masked

# Gemma 모델로 실행 (마스킹 정보 없음)
echo "================================================="
echo "            Gemma 모델 실행 중 (마스킹 정보 없음)           "
echo "================================================="
python generate_mitigated_dialect.py \
    --input_file "./masked_data.json" \
    --output_dir "./results" \
    --model_names "princeton-nlp/gemma-2-9b-it-SimPO"

# RoBERTa 모델로 실행 (마스킹 정보 있음)
echo "================================================="
echo "            RoBERTa 모델 실행 중 (마스킹 정보 있음)           "
echo "================================================="
python generate_mitigated_dialect.py \
    --input_file "./masked_data.json" \
    --output_dir "./results" \
    --model_names "FacebookAI/roberta-base" \
    --use_masked

# RoBERTa 모델로 실행 (마스킹 정보 없음)
echo "================================================="
echo "            RoBERTa 모델 실행 중 (마스킹 정보 없음)           "
echo "================================================="
python generate_mitigated_dialect.py \
    --input_file "./masked_data.json" \
    --output_dir "./results" \
    --model_names "FacebookAI/roberta-base"

echo "================================================="
echo "            방언 완화 생성 완료           "
echo "=================================================" 