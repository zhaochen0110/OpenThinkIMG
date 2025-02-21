source ~/.bashrc
source ~/anaconda3/bin/activate r1-v

# environment variables
export OMP_NUM_THREADS=4
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/songmingyang/code/reasoning/R1-V/src/r1-v/src/open_r1
cd $code_base
job_id=4294232
export SLURM_JOB_ID=${job_id}
unset SLURM_JOB_ID

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="/mnt/petrelfs/songmingyang/code/reasoning/R1-V/src/r1-v/local_scripts/scripts/logs/debug_log_2b.log"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export WANDB_PROJECT="r1_v"

gpus=0
cpus=2
quotatype="reserved"
CUDA_VISIBLE_DEVICES="4,5,6,7"
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-37 \
torchrun --nproc_per_node="3" \
--nnodes="1" \
--node_rank="0" \
--master_addr="127.0.0.1" \
--master_port="15246" \
./grpo.py \
--output_dir /mnt/petrelfs/songmingyang/code/reasoning/R1-V/src/r1-v/src/open_r1/scripts/outputs \
--model_name_or_path /mnt/petrelfs/songmingyang/songmingyang/model/mm/Qwen2-VL-2B-Instruct \
--dataset_name /mnt/petrelfs/songmingyang/songmingyang/data/mm/annotation/clevr_cogen_a_train \
--deepspeed /mnt/petrelfs/songmingyang/code/reasoning/R1-V/src/r1-v/local_scripts/zero3.json \
--max_prompt_length 512 \
--max_completion_length 512 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--logging_steps 1 \
--bf16 \
--report_to wandb \
--gradient_checkpointing false \
--attn_implementation flash_attention_2 \
--max_pixels 401408 \
--num_train_epochs 1 \
--run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
--save_steps 100 \
--save_only_model true \
--use_vllm \
--vllm_device "auto" \
--num_generations 3   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

# srun --partition=MoE --mpi=pmi2 --nodes=1 -c 2 -w SH-IDCA1404-10-140-54-37 --gres=gpu:0 --quotatype="reserved" jupyter lab  --ip=0.0.0.0 --port=10049

