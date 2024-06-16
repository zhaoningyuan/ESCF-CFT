#!/bin/bash

datasets=("simp" "emdg" "inqqg" "exp" "hgen")
model_name="bloomz-1b1"
base_model_path="finetuned_models/${model_name}"
# init model save path

for dataset in "${datasets[@]}"; do
    # eval on base model
    echo "now evaluating base model on ${dataset}"
    deepspeed --num_gpus=2 run.py \
        --model_name_or_path="${base_model_path}" \
        --output_dir="${base_model_path}/${dataset}/results" \
        --overwrite_output_dir \
        --dataset_path="dataset/train/${dataset}/test.jsonl" \
        --cutoff_len=512 \
        --val_size=0 \
        --do_predict \
        --report_to="none" \
        --per_device_eval_batch_size=8 \
        --predict_with_generate \
        --bf16
done