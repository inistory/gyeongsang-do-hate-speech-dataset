#!/bin/bash

# 현재 스크립트의 디렉토리 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#training
python detection_curriculum_training.py \
    --model_name_or_path EleutherAI/polyglot-ko-5.8b \
    --train_file "${SCRIPT_DIR}/detection_easy.jsonl" \
    --validation_file "${SCRIPT_DIR}/gs_kold_valid.json" \
    --test_file "${SCRIPT_DIR}/gs_kold_test.json" \
    --do_train \
    --output_dir "${SCRIPT_DIR}/output_easy" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01


#evaluation
python detection_curriculum_training.py \
    --model_name_or_path "${SCRIPT_DIR}/output_easy" \
    --validation_file "${SCRIPT_DIR}/gs_kold_valid.json" \
    --test_file "${SCRIPT_DIR}/gs_kold_test.json" \
    --do_eval \
    --output_dir "${SCRIPT_DIR}/output_eval" \
    --trust_remote_code true \
    --base_model_name_or_path "EleutherAI/polyglot-ko-5.8b"

#prediction
python detection_curriculum_training.py \
    --model_name_or_path "${SCRIPT_DIR}/output_easy" \
    --test_file "${SCRIPT_DIR}/gs_kold_test.json" \
    --do_predict \
    --output_dir "${SCRIPT_DIR}/output_predict" \
    --trust_remote_code true \
    --base_model_name_or_path "EleutherAI/polyglot-ko-5.8b"