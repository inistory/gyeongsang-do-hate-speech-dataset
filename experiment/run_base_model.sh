#!/bin/bash

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 모델 및 데이터 경로 설정
MODEL_PATHS=(
    "nlpai-lab/kullm-polyglot-12.8b-v2"
    "EleutherAI/polyglot-ko-5.8b"
)
TEST_FILE="./gs_kold_test.json" # 테스트 데이터 경로 수정

# 각 모델에 대해 실행
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    # 모델 이름에서 디렉토리 이름 생성
    MODEL_DIR_NAME=$(echo $MODEL_PATH | sed 's/\//_/g')
    OUTPUT_DIR="./output_base_${MODEL_DIR_NAME}_predict"
    
    echo "모델 $MODEL_PATH 학습을 시작합니다..."
    
    # 출력 디렉토리 생성
    mkdir -p $OUTPUT_DIR
    
    # 실행 명령어
    python detection_base_model.py \
        --model_name_or_path $MODEL_PATH \
        --test_file $TEST_FILE \
        --max_seq_length 128 \
        --output_dir $OUTPUT_DIR \
        --do_predict True \
        --no_cuda False \
        --dataloader_pin_memory False \
        --trust_remote_code True \
        2>&1 | tee $OUTPUT_DIR/run.log
    
    echo "모델 $MODEL_PATH 학습이 완료되었습니다. 결과는 $OUTPUT_DIR 디렉토리에 저장되었습니다."
    echo "----------------------------------------"
done

echo "모든 모델의 학습이 완료되었습니다." 