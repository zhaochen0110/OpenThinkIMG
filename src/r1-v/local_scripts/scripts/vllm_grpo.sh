
source ~/.bashrc
source ~/anaconda3/bin/activate r1-v

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

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/songmingyang/code/reasoning/R1-V/src/r1-v/src/open_r1
cd $code_base
job_id=4294232
export SLURM_JOB_ID=${job_id}
unset SLURM_JOB_ID

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="/mnt/petrelfs/songmingyang/code/reasoning/R1-V/src/r1-v/local_scripts/scripts/logs/debug_log_2b.log"
# export CUDA_VISIBLE_DEVICES="2,3,7"
export WANDB_PROJECT="r1_v"
export RUN_NAME="Qwen2-VL-2B-GRPO-CHARTGEMMA-1000"

model_path=/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/tool_sft/tool_sft_2000_replace_Qwen2-VL
# model_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/Qwen2-VL-2B-Instruct
model_path=/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/tool_sft/tool_sft_2B_2000_replace_Qwen2-VL

data_path=/mnt/petrelfs/songmingyang/songmingyang/runs/tool_factory/chart_data/chartgemma_1000.json
data_path=/mnt/petrelfs/share_data/suzhaochen/datasets/chartgemma_cot/split_train.json

gpus=0
cpus=2
quotatype="reserved"
CUDA_VISIBLE_DEVICES="3,4,5,6,7" srun --partition=MoE --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-5 \
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ./tool_grpo.py --use_vllm True \
    --output_dir /mnt/petrelfs/songmingyang/code/reasoning/R1-V/src/r1-v/src/open_r1/scripts/outputs \
    --model_name_or_path ${model_path} \
    --dataset_name ${data_path} \
    --max_prompt_length 1024 \
    --max_completion_length 2048 \
    --temperature 1.0 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 400000 \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model true \
    --use_tool