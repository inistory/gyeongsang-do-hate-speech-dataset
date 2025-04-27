#!/bin/bash



#####[Curriculum Learning]
#training
echo "================================================="
echo "            CURRICULUM LEARNING TRAINING           "
echo "================================================="
CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
    --model_name_or_path EleutherAI/polyglot-ko-5.8b \
    --train_files "./detection_easy.jsonl" "./detection_medium.jsonl" "./detection_hard.jsonl" \
    --validation_file "./gs_kold_valid.json" \
    --test_file "./gs_kold_test.json" \
    --do_train True \
    --output_dir "./output_curriculum" \
    --curriculum_epochs 10 10 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.1 \
    --fp16 False \
    --gradient_accumulation_steps 2 \
    --trust_remote_code True

#evaluation
echo "================================================="
echo "                  EVALUATION                       "
echo "================================================="
CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
    --model_name_or_path "./output_curriculum" \
    --validation_file "./gs_kold_valid.json" \
    --test_file "./gs_kold_test.json" \
    --do_eval \
    --do_train False \
    --output_dir "./output_curriculum_eval" \
    --trust_remote_code true \
    --base_model_name_or_path "EleutherAI/polyglot-ko-5.8b"

#prediction
echo "================================================="
echo "                  PREDICTION                       "
echo "================================================="
CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
    --model_name_or_path "./output_curriculum" \
    --test_file "./gs_kold_test.json" \
    --do_predict \
    --do_train False \
    --output_dir "./output_curriculum_predict" \
    --trust_remote_code true \
    --base_model_name_or_path "EleutherAI/polyglot-ko-5.8b"