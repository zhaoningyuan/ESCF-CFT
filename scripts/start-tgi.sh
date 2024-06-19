# choose from "bloomz-1b1", "bloomz-1b1->simp", ...
model_path="bloomz-1b1"
docker run \
    --name "${model_path%%-*}" \
    -d \
    --gpus all  \
    --shm-size 1g \
    -p 8181:80 \
    -v "${PWD}/finetuned_models/:/data" \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id "/data/${model_path}"