export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

source activate newopenr1
cd /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/r1-v/

job_id=4294232
export SLURM_JOB_ID=${job_id}
unset SLURM_JOB_ID

export RUN_NAME="Qwen2.5-VL-3B-chartgemma-reachqa-2epoch-sft-new"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,7"
hostname="SH-IDCA1404-10-140-54-36"
# data_path=/mnt/petrelfs/share_data/suzhaochen/datasets/chargemma_final/select/chartgemma-reachqa-combined-sharegpt.json
data_path=/mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/r1-v/test.json
# data_path=/mnt/petrelfs/share_data/suzhaochen/datasets/chargemma_final/select/chartgemma-combined-sharegpt.json
# data_path=HuggingFaceH4/Bespoke-Stratos-17k
model_path=/mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2.5-VL-3B-Instruct

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
rm -rf /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/r1-v/local_scripts/logs/$RUN_NAME.log
export LOG_PATH="/mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/r1-v/local_scripts/logs/${RUN_NAME}.log"
export WANDB_PROJECT="r1_v"

master_port=$(shuf -i 10000-65535 -n 1) 
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
nproc_per_node=$((num_gpus))

gpus=0
cpus=2
quotatype="reserved"

srun --partition=MoE --job-name="sft" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=${quotatype} \
-w ${hostname} \
    accelerate launch --num_machines 1 --num_processes 6 --main_process_port 29502 --multi_gpu\
    src/open_r1/sft.py \
    --output_dir /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/R1-V/src/output_path/$RUN_NAME \
    --model_name_or_path ${model_path} \
    --dataset_name ${data_path} \
    --seed 42 \
    --learning_rate 2e-5 \
    --max_seq_length 4096 \
    --deepspeed /mnt/petrelfs/share_data/suzhaochen/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --bf16 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 90 \
    --warmup_ratio 0.1 \
    --save_only_model true