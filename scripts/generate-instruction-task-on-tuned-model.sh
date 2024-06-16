#!/bin/bash

datasets=("simp" "emdg" "inqqg" "exp" "hgen")
model_name="bloomz-1b1"
# init model save path
last_model_path="finetuned_models/${model_name}"

for dataset in "${datasets[@]}"; do
    next_model_path="${last_model_path}->${dataset}"
    # Eval stage
    echo "now evaluating tuned model on ${dataset}"
    deepspeed --num_gpus=2 run.py \
        --model_name_or_path="${next_model_path}" \
        --output_dir="${next_model_path}/results" \
        --overwrite_output_dir \
        --dataset_path="dataset/train/${dataset}/test.jsonl" \
        --cutoff_len=512 \
        --val_size=0 \
        --do_predict \
        --report_to="none" \
        --per_device_eval_batch_size=8 \
        --predict_with_generate \
        --bf16

    # Update last_model_path for next iteration
    last_model_path="${next_model_path}"
done