path="finetuned_models/bloomz-1b1"
declare -A tasks_map
tasks_map["reasoning"]="piqa,boolq,winogrande,hellaswag,mathqa,mutual"
tasks_map["domain-knowledge"]="mmlu"
tasks_map["reading-comprehension"]="race"


# args 
model_dir="finetuned_models/"
model_name="bloomz-1b1"
device="cuda:1"
task_type="general-tasks"
# sub_task_type="reasoning", "domain-knowledge", "reading-comprehension"
sub_task_type="reasoning"

# start script
tasks=${tasks_map[$sub_task_type]}
if [ "$tasks" = "mmlu" ]; then
    num_fewshot=5
else
    num_fewshot=0
fi
folders=($(find "$model_dir" -type d -name "${model_name}*"))
for folder in "${folders[@]}"; do
    instruc_task_name="${folder##*->}"
    if [ "$instruc_task_name" = "$folder" ]; then
        instruc_task_name="base"
    fi
    echo "Model: $instruc_task_name"
    lm_eval \
        --model hf \
        --model_args pretrained=${folder} \
        --tasks  $tasks \
        --device $device \
        --output_path "results/${task_type}/${sub_task_type}/${instruc_task_name}" \
        --batch_size "auto" \
        --num_fewshot $num_fewshot
done
    