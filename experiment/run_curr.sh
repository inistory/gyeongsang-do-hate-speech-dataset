#!/bin/bash
mkdir offload
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 모델 리스트 정의
MODELS=(
    "EleutherAI/polyglot-ko-5.8b"
    "nlpai-lab/kullm-polyglot-12.8b-v2"
)

# 각 모델에 대해 실험 실행
for MODEL in "${MODELS[@]}"; do
    # 모델 이름에서 마지막 부분만 추출
    MODEL_NAME=$(echo $MODEL | awk -F'/' '{print $NF}')
    
    echo "================================================="
    echo "            Curriculum Learning with $MODEL_NAME           "
    echo "================================================="
    
    # 학습
    CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
        --model_name_or_path "$MODEL" \
        --train_files "./detection_easy.jsonl" "./detection_medium.jsonl" "./detection_hard.jsonl" \
        --validation_file "./gs_kold_valid.json" \
        --test_file "./gs_kold_test.json" \
        --do_train True \
        --output_dir "./output_curriculum_${MODEL_NAME}" \
        --curriculum_epochs 1 1 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --learning_rate 1e-4 \
        --weight_decay 0.01 \
        --max_grad_norm 0.5 \
        --warmup_ratio 0.1 \
        --fp16 False \
        --gradient_accumulation_steps 4 \
        --trust_remote_code True

    # 평가
    echo "================================================="
    echo "            Evaluating $MODEL_NAME           "
    echo "================================================="
    CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
        --model_name_or_path "./output_curriculum_${MODEL_NAME}" \
        --validation_file "./gs_kold_valid.json" \
        --test_file "./gs_kold_test.json" \
        --do_eval \
        --do_train False \
        --output_dir "./output_curriculum_${MODEL_NAME}_eval" \
        --trust_remote_code true \
        --base_model_name_or_path "$MODEL"

    # 예측
    echo "================================================="
    echo "            Predicting with $MODEL_NAME           "
    echo "================================================="
    CUDA_VISIBLE_DEVICES=0 python detection_curriculum_training.py \
        --model_name_or_path "./output_curriculum_${MODEL_NAME}" \
        --test_file "./gs_kold_test.json" \
        --do_predict \
        --do_train False \
        --output_dir "./output_curriculum_${MODEL_NAME}_predict" \
        --trust_remote_code true \
        --base_model_name_or_path "$MODEL"
done


# #full-finetuning
# echo "================================================="
# echo "            FULL-FINETUNING         "
# echo "================================================="
# # 모델 리스트 정의
# MODELS=(
#     "EleutherAI/polyglot-ko-5.8b"
#     "nlpai-lab/kullm-3-7b"
#     "beomi/KoAlpaca-Polyglot-12.8B"
# )

# # 각 모델에 대해 실험 실행
# for MODEL in "${MODELS[@]}"; do
#     # 모델 이름에서 마지막 부분만 추출
#     MODEL_NAME=$(echo $MODEL | awk -F'/' '{print $NF}' | sed 's/[^a-zA-Z0-9]//g')
    
#     echo "================================================="
#     echo "            Training with $MODEL_NAME           "
#     echo "================================================="
    
#     # 학습
#     CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
#         --model_name_or_path "$MODEL" \
#         --train_file "./detection_full.jsonl" \
#         --validation_file "./gs_kold_valid.json" \
#         --test_file "./gs_kold_test.json" \
#         --do_train True \
#         --output_dir "./output_full_${MODEL_NAME}" \
#         --per_device_train_batch_size 8 \
#         --per_device_eval_batch_size 8 \
#         --learning_rate 1e-4 \
#         --weight_decay 0.01 \
#         --max_grad_norm 0.5 \
#         --warmup_ratio 0.1 \
#         --fp16 False \
#         --gradient_accumulation_steps 4 \
#         --trust_remote_code True \
#         --num_train_epochs 5

#     # 평가
#     echo "================================================="
#     echo "            Evaluating $MODEL_NAME           "
#     echo "================================================="
#     CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
#         --model_name_or_path "./output_full_${MODEL_NAME}" \
#         --validation_file "./gs_kold_valid.json" \
#         --test_file "./gs_kold_test.json" \
#         --do_eval \
#         --do_train False \
#         --output_dir "./output_full_${MODEL_NAME}_eval" \
#         --trust_remote_code true \
#         --base_model_name_or_path "$MODEL"

#     # 예측
#     echo "================================================="
#     echo "            Predicting with $MODEL_NAME           "
#     echo "================================================="
#     CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
#         --model_name_or_path "./output_full_${MODEL_NAME}" \
#         --test_file "./gs_kold_test.json" \
#         --do_predict \
#         --do_train False \
#         --output_dir "./output_full_${MODEL_NAME}_predict" \
#         --trust_remote_code true \
#         --base_model_name_or_path "$MODEL"
# done


# #hybrid-curriculum


# # 모델 리스트 정의
# MODELS=(
#     "beomi/KoAlpaca-Polyglot-12.8B"
#     "EleutherAI/polyglot-ko-5.8b"
#     "nlpai-lab/kullm-polyglot-12.8b-v2"
# )

# # 각 모델에 대해 실험 실행
# for MODEL in "${MODELS[@]}"; do
#     # 모델 이름에서 마지막 부분만 추출
#     MODEL_NAME=$(echo $MODEL | awk -F'/' '{print $NF}')
    
#     echo "================================================="
#     echo "            Training with $MODEL_NAME           "
#     echo "================================================="
    
#     # 학습
#     CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
#         --model_name_or_path "$MODEL" \
#         --train_file "./detection_full.jsonl" \
#         --validation_file "./gs_kold_valid.json" \
#         --test_file "./gs_kold_test.json" \
#         --do_train True \
#         --output_dir "./output_full_${MODEL_NAME}" \
#         --per_device_train_batch_size 16 \
#         --per_device_eval_batch_size 16 \
#         --learning_rate 5e-5 \
#         --weight_decay 0.01 \
#         --max_grad_norm 0.5 \
#         --warmup_ratio 0.1 \
#         --fp16 False \
#         --gradient_accumulation_steps 2 \
#         --trust_remote_code True \
#         --num_train_epochs 5

#     # 평가
#     echo "================================================="
#     echo "            Evaluating $MODEL_NAME           "
#     echo "================================================="
#     CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
#         --model_name_or_path "./output_full_${MODEL_NAME}" \
#         --validation_file "./gs_kold_valid.json" \
#         --test_file "./gs_kold_test.json" \
#         --do_eval \
#         --do_train False \
#         --output_dir "./output_full_${MODEL_NAME}_eval" \
#         --trust_remote_code true \
#         --base_model_name_or_path "$MODEL"

#     # 예측
#     echo "================================================="
#     echo "            Predicting with $MODEL_NAME           "
#     echo "================================================="
#     CUDA_VISIBLE_DEVICES=0 python detection_full_finetuning.py \
#         --model_name_or_path "./output_full_${MODEL_NAME}" \
#         --test_file "./gs_kold_test.json" \
#         --do_predict \
#         --do_train False \
#         --output_dir "./output_full_${MODEL_NAME}_predict" \
#         --trust_remote_code true \
#         --base_model_name_or_path "$MODEL"
# done