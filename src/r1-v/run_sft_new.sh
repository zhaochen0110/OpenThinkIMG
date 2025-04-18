#!/bin/bash
# 取消HF镜像设置（根据实际需求决定是否保留）
#export HF_ENDPOINT=https://hf-mirror.com
#unset HF_ENDPOINT

# 环境初始化
source activate newopenr1
cd /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/r1-v/

# 训练配置
export RUN_NAME="Qwen2.5-VL-3B-chartgemma-reachqa-2epoch-sft"
export WANDB_PROJECT="r1_v"
export LOG_PATH="./local_scripts/logs/${RUN_NAME}.log"
export NCCL_DEBUG=INFO  # 启用NCCL调试信息
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网卡名称修改

# 路径配置
data_path="/mnt/petrelfs/share_data/suzhaochen/datasets/chargemma_final/select/chartgemma-reachqa-combined-sharegpt.json"
model_path="/mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2.5-VL-3B-Instruct"
output_dir="./output_path/$RUN_NAME"
log_dir="./local_scripts/logs"

# 创建必要目录
mkdir -p ${output_dir} ${log_dir}
rm -rf ${log_dir}/${RUN_NAME}.log

# 分布式配置
master_addr="SH-IDCA1404-10-140-54-5"
master_port=$(shuf -i 10000-65535 -n 1)
num_nodes=2
gpus_per_node=8
total_gpus=$((num_nodes * gpus_per_node))
node_list="SH-IDCA1404-10-140-54-5,SH-IDCA1404-10-140-54-80"

# SLURM参数优化
srun --partition=MoE --job-name="sft" --mpi=pmi2 --gres=gpu:0 --nodes=${num_nodes} --ntasks-per-node=1 --cpus-per-task=$((gpus_per_node * 8)) --kill-on-bad-exit=1 --quotatype=reserved -w ${node_list} \
    accelerate launch --num_processes ${total_gpus} --num_machines ${num_nodes} --main_process_ip ${master_addr} --main_process_port ${master_port} --machine_rank \$SLURM_NODEID \  
    src/open_r1/sft.py \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_path} \
    --dataset_name ${data_path} \
    --seed 42 \
    --learning_rate 2e-5 \
    --max_seq_length 4096 \
    --deepspeed ./local_scripts/zero3_offload.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --bf16 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 312313 \
    --warmup_ratio 0.1 \
    --save_only_model 