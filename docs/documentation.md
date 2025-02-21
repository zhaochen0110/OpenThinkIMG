
# Multimodal RLHF with Tool Calling

## Requirements
`tool_server` is required if you want to call tools.
`R1-V` is required if you want to use multimodal GRPO.

## Usage
```bash
git clone https://github.com/ssmisya/R1-V.git
cd R1-V/src/r1-v/src/open_r1
# Install the required packages shown in README.md

source ~/.bashrc
source ~/anaconda3/bin/activate r1-v


export DEBUG_MODE="true" # or "false"
export LOG_PATH="../../local_scripts/scripts/logs/debug_log_2b.log"
export WANDB_PROJECT="r1_v"
export RUN_NAME="Qwen2-VL-2B-GRPO-CLEVR-70k"


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ./grpo.py --use_vllm True \
    --output_dir ../../local_scripts/scripts/outputs \
    --model_name_or_path /mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2-VL-2B-Instruct \
    --dataset_name /mnt/petrelfs/share_data/songmingyang/data/mm/annotation/clevr_cogen_a_train \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
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

```

An example run script can be found at: `R1-V/src/r1-v/local_scripts/scripts/vllm_grpo.sh`