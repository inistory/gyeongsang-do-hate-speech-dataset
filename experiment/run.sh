#!/bin/bash

#training
echo "================================================="
echo "                   TRAINING                        "
echo "================================================="
CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
    --model_name_or_path EleutherAI/polyglot-ko-5.8b \
    --train_file "./gs_kold_train.json" \
    --validation_file "./gs_kold_valid.json" \
    --test_file "./gs_kold_test.json" \
    --do_train True\
    --output_dir "./output_easy" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01


#evaluation
echo "================================================="
echo "                  EVALUATION                       "
echo "================================================="
CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
    --model_name_or_path "./output_easy" \
    --validation_file "./gs_kold_valid.json" \
    --test_file "./gs_kold_test.json" \
    --do_eval \
    --do_train False \
    --output_dir "./output_eval" \
    --trust_remote_code true \
    --base_model_name_or_path "EleutherAI/polyglot-ko-5.8b"

#prediction

echo "================================================="
echo "                  PREDICTION                       "
echo "================================================="
CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
    --model_name_or_path "./output_easy" \
    --test_file "./gs_kold_test.json" \
    --do_predict \
    --do_train False \
    --output_dir "./output_predict" \
    --trust_remote_code true \
    --base_model_name_or_path "EleutherAI/polyglot-ko-5.8b"