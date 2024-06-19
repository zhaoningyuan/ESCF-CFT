# args
task_type="instruction-tasks"
declare -A metrics_map
metrics_map["simp"]="SARI"
metrics_map["emdg"]="BLEU"
metrics_map["inqqg"]="BLEU"
metrics_map["exp"]="BLEU"
metrics_map["hgen"]="ROUGE"


# start script
task_dirs=($(find "results/${task_type}/" -mindepth 1 -type d))
for task_dir in "${task_dirs[@]}"; do
    task="${task_dir##*/}"
    echo "Task: $task"
    # eval on tuned model
    python eval.py \
        --metric ${metrics_map[$task]} \
        --task $task \
        --model_type "tuned" \
        --pred_file_path "results/${task_type}/${task}/tuned_model_predictions.jsonl" \
        --ref_file_path "dataset/train/${task}/test.jsonl" \
        --output_path "results/${task_type}/results.jsonl" \
        --special_column "normal_sentence" \
        --src_column "prompt" \
        --pred_column "pred" \
        --ref_column "reference"
    # eval on base model
    python eval.py \
        --metric ${metrics_map[$task]} \
        --task $task \
        --model_type "base" \
        --pred_file_path "results/${task_type}/${task}/base_model_predictions.jsonl" \
        --ref_file_path "dataset/train/${task}/test.jsonl" \
        --output_path "results/${task_type}/results.jsonl" \
        --special_column "normal_sentence" \
        --src_column "prompt" \
        --pred_column "pred" \
        --ref_column "reference"
done