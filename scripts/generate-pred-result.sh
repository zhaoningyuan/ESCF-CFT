# args 
model_dir="finetuned_models/"
model_name="bloomz-1b1"
port=8181
wait_container_time=15
batch_size=64
max_new_tokens=256
task_type="instruction-tasks"


# start script
folders=($(find "$model_dir" -type d -name "${model_name}*"))
declare -a tasks
echo "generate predictions for $model_name on tuned models"
for folder in "${folders[@]}"; do
    model_folder="${folder##*/}"
    task="${folder##*->}"
    # generate on tuned model
    if [ "$task" != "$folder" ]; then
        echo "Task: $task"
        echo "Model: $model_folder"
        tasks+=("$task")
        # start tgi container
        echo "Starting container for $model_folder"
        docker run \
            --name "${model_name%%-*}" \
            -d \
            --gpus all  \
            --shm-size 1g \
            -p $port:80 \
            -v "${PWD}/finetuned_models/:/data" \
            ghcr.io/huggingface/text-generation-inference:latest \
            --model-id "/data/${model_folder}"
        # wait for tgi container to start
        sleep $wait_container_time
        echo "Container started"
        # generate predictions
        python generate.py \
            --input_file "${PWD}/dataset/train/$task/test.jsonl" \
            --output_file "results/${task_type}/${task}/tuned_model_predictions.jsonl" \
            --batch_size $batch_size \
            --max_new_tokens $max_new_tokens
        # stop and remove tgi container
        echo "Stopping container for $model_folder"
        docker stop "${model_name%%-*}"
        echo "Container stopped"
        docker rm "${model_name%%-*}"
        ecjo "Container removed"
    fi
done

echo "generate predictions for $model_name on base model"
for folder in "${folders[@]}"; do
    task="${folder##*->}"
    if [ "$task" == "$folder" ]; then
        model_folder="${folder##*/}"
        # start tgi container
        echo "Starting container for $model_folder"
        docker run \
            --name "${model_name%%-*}" \
            -d \
            --gpus all  \
            --shm-size 1g \
            -p $port:80 \
            -v "${PWD}/finetuned_models/:/data" \
            ghcr.io/huggingface/text-generation-inference:latest \
            --model-id "/data/${model_folder}"
        # wait for tgi container to start
        sleep $wait_container_time
        echo "Container started"
        for task in "${tasks[@]}"; do
            echo "Task: $task"
            python generate.py \
                --input_file "${PWD}/dataset/train/$task/test.jsonl" \
                --output_file "results/${task_type}/${task}/base_model_predictions.jsonl" \
                --batch_size $batch_size \
                --max_new_tokens $max_new_tokens
        done
        # stop and remove tgi container
        echo "Stopping container for $model_folder"
        docker stop "${model_name%%-*}"
        echo "Container stopped"
        docker rm "${model_name%%-*}"
        echo "Container removed"
    fi
done

# python generate.py \
    # --input_file "${PWD}/dataset/train/simp/test.jsonl" \
    # --output_file "${PWD}/finetuned_models/bloomz-1b1->simp/results/generated_predictions.jsonl" \
    # --batch_size 64 \
    # --max_new_tokens 256 