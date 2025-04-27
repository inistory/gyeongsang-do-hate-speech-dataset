#!/bin/bash



#####[full Learning]
#training
echo "================================================="
echo "            full LEARNING TRAINING           "
echo "================================================="
CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
    --model_name_or_path EleutherAI/polyglot-ko-5.8b \
    --train_file "./detection_full.jsonl" \
    --validation_file "./gs_kold_valid.json" \
    --test_file "./gs_kold_test.json" \
    --do_train True \
    --output_dir "./output_full" \
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
CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
    --model_name_or_path "./output_full" \
    --validation_file "./gs_kold_valid.json" \
    --test_file "./gs_kold_test.json" \
    --do_eval \
    --do_train False \
    --output_dir "./output_full_eval" \
    --trust_remote_code true \
    --base_model_name_or_path "EleutherAI/polyglot-ko-5.8b"

#prediction
echo "================================================="
echo "                  PREDICTION                       "
echo "================================================="
CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
    --model_name_or_path "./output_full" \
    --test_file "./gs_kold_test.json" \
    --do_predict \
    --do_train False \
    --output_dir "./output_full_predict" \
    --trust_remote_code true \
    --base_model_name_or_path "EleutherAI/polyglot-ko-5.8b"