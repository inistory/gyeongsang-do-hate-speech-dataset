#!/bin/bash

#full-finetuning
echo "================================================="
echo "            FULL-FINETUNING         "
echo "================================================="
# 모델 리스트 정의
MODELS=(
    # "EleutherAI/polyglot-ko-5.8b"
    "beomi/KoAlpaca-Polyglot-12.8B"
    # "nlpai-lab/kullm-3-7b"
)
# 각 모델에 대해 실험 실행
for MODEL in "${MODELS[@]}"; do
    # 모델 이름에서 마지막 부분만 추출
    MODEL_NAME=$(echo $MODEL | awk -F'/' '{print $NF}')
    
    echo "================================================="
    echo "            Training with $MODEL_NAME           "
    echo "================================================="
    
    # 학습
    CUDA_VISIBLE_DEVICES=1 python detection_full_finetuning.py \
        --model_name_or_path "$MODEL" \
        --train_file "./detection_full.jsonl" \
        --validation_file "./gs_kold_valid.json" \
        --test_file "./gs_kold_test.json" \
        --do_train True \
        --output_dir "./output_full_${MODEL_NAME}" \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --learning_rate 1e-5 \
        --weight_decay 0.05 \
        --max_grad_norm 0.5 \
        --warmup_ratio 0.2 \
        --fp16 False \
        --bf16 True \
        --gradient_accumulation_steps 16 \
        --trust_remote_code True \
        --num_train_epochs 10

    # 평가
    echo "================================================="
    echo "            Evaluating $MODEL_NAME           "
    echo "================================================="
    CUDA_VISIBLE_DEVICES=1 python detection_full_finetuning.py \
        --model_name_or_path "./output_full_${MODEL_NAME}" \
        --validation_file "./gs_kold_valid.json" \
        --test_file "./gs_kold_test.json" \
        --do_eval True \
        --do_train False \
        --output_dir "./output_full_${MODEL_NAME}_eval" \
        --trust_remote_code True \
        --base_model_name_or_path "$MODEL" \
        --save_strategy "no" \
        --evaluation_strategy "no"

    # 예측
    echo "================================================="
    echo "            Predicting with $MODEL_NAME           "
    echo "================================================="
    CUDA_VISIBLE_DEVICES=1 python detection_full_finetuning.py \
        --model_name_or_path "./output_full_${MODEL_NAME}" \
        --test_file "./gs_kold_test.json" \
        --do_predict True \
        --do_train False \
        --output_dir "./output_full_${MODEL_NAME}_predict" \
        --trust_remote_code True \
        --base_model_name_or_path "$MODEL" \
        --save_strategy "no" \
        --evaluation_strategy "no"
done
