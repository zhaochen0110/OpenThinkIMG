source activate openr1
cd /mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

QWEN_PATH="/mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2-VL-2B-Instruct "
# HF_DATASET="/mnt/petrelfs/share_data/songmingyang/data/mm/annotation/clevr_cogen_a_train" 
OUTPUT_DIR="/mnt/petrelfs/share_data/suzhaochen/r1/R1-V-tool/src/output_path/chartgemma_default" 
RUN_NAME="Qwen2-VL-2B-chartgemmadefault"
HF_DATASET="/mnt/petrelfs/share_data/suzhaochen/datasets/chartgemma_cot/split_train.json"

quotatype="reserved"
jobname="r1v"
srun --partition=MoE --job-name=${jobname} --job-name="eval" --mpi=pmi2  --gres=gpu:0 -w SH-IDCA1404-10-140-54-81 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=${quotatype}  \
torchrun --nproc_per_node="3" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12321" \
    src/open_r1/tool_grpo.py --use_vllm True \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $HF_DATASET \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --num_generations 7 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 500000 \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true