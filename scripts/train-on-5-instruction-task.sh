#!/bin/bash

datasets=("simp" "emdg" "inqqg" "exp" "hgen")
model_name="bloomz-1b1"
# init model save path
last_model_path="finetuned_models/${model_name}"

for dataset in "${datasets[@]}"; do
    next_model_path="${last_model_path}->${dataset}"
    echo "now training on ${dataset}"
    # Train stage
    deepspeed --num_gpus=2 run.py \
        --model_name_or_path="${last_model_path}" \
        --output_dir="${next_model_path}" \
        --overwrite_output_dir \
        --dataset_path="dataset/train/${dataset}/train.jsonl" \
        --cutoff_len=512 \
        --overwrite_cache \
        --val_size=0 \
        --finetuning_type="full" \
        --report_to="wandb" \
        --deepspeed "./configs/ds_z1.json" \
        --learning_rate=2e-5 \
        --per_device_train_batch_size=2 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=8 \
        --logging_steps=10 \
        --num_train_epochs=3 \
        --max_samples=10000 \
        --max_steps=-1 \
        --gradient_checkpointing \
        --bf16 \
        --do_train

    # Update last_model_path for next iteration
    last_model_path="${next_model_path}"
done