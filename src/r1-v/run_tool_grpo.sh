# environment variables
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address
# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY

# curl -X POST http://SH-IDCA1404-10-140-54-5:20001/list_models
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=18000
export NCCL_LAUNCH_MODE=GROUP
export CUDA_LAUNCH_BLOCKING=1
export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

source activate openr1
cd /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/r1-v/
job_id=4294232
export SLURM_JOB_ID=${job_id}
unset SLURM_JOB_ID

# nohup bash run_tool_grpo.sh > logs/Qwen2-VL-2B-grpo-chartgemma-katrina-noformat-tool-batch144.log &
############################
# u need to revise
export RUN_NAME="Qwen2-VL-2B-grpo-chartgemma-chartgemma-katrina-noformat-batch144"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
# export CUDA_VISIBLE_DEVICES="0,1,4,6"
hostname="SH-IDCA1404-10-140-54-16" 
# data_path=/mnt/petrelfs/share_data/suzhaochen/datasets/reachqa_final/reachqa_chartgemma_data.json
data_path=/mnt/petrelfs/share_data/suzhaochen/datasets/chargemma_final/chartgemma_rl.json
model_path=/mnt/petrelfs/share_data/suzhaochen/LLaMA-Factory/saves/Qwen2-VL-chartgemma-combine
# model_path=/mnt/petrelfs/share_data/suzhaochen/LLaMA-Factory/saves/Qwen2-VL-chartgemma-reachqa-combined
# /mnt/petrelfs/share_data/suzhaochen/LLaMA-Factory/saves/Qwen2-VL-chartgemma-reachqa-cota
# 
###########################
# onlytool: /mnt/petrelfs/share_data/suzhaochen/LLaMA-Factory/saves/Qwen2-VL-chartgemma-onlytool
# combine: /mnt/petrelfs/share_data/suzhaochen/LLaMA-Factory/saves/Qwen2-VL-chartgemma-combine
# default: /mnt/petrelfs/share_data/suzhaochen/LLaMA-Factory/saves/Qwen2-VL-chartgemma-tool

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
rm -rf /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/r1-v/local_scripts/logs/$RUN_NAME.log
export LOG_PATH="/mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/r1-v/local_scripts/logs/${RUN_NAME}.log"
export WANDB_PROJECT="r1_v"

master_port=$(shuf -i 10000-65535 -n 1) 
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
nproc_per_node=$((num_gpus - 1))


# data_path=/mnt/petrelfs/share_data/suzhaochen/datasets/reachqa_final/reachqa_chartgemma_data.json

gpus=0
cpus=2
quotatype="reserved"
srun --partition=MoE --job-name="katrina" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=${quotatype}  \
-w ${hostname} \
torchrun --nproc_per_node=${nproc_per_node} \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port=${master_port} \
    src/open_r1/tool_grpo.py --use_vllm True \
    --output_dir /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/output_path/$RUN_NAME \
    --model_name_or_path ${model_path} \
    --dataset_name ${data_path} \
    --max_prompt_length 1024 \
    --max_completion_length 3500 \
    --temperature 1.0 \
    --seed 42 \
    --learning_rate 1e-6 \
    --num_generations 12 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 12 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 400000 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 200 \
    --save_only_model true \
    --controller_addr http://SH-IDCA1404-10-140-54-16:20001 \
    --use_tool

#     --max_steps 10000 \