

source ~/.bashrc
source ~/anaconda3/bin/activate smoe

export SLURM_JOB_ID=3873423 
unset SLURM_JOB_ID      

controller_addr=$1


code_base=/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/inferencer
cd $code_base

gpus=0
cpus=16
quotatype="auto"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="inference" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./singleloop_inference.py \
--input_path /mnt/petrelfs/share_data/mmtool/CharXiv/code-CharXiv/data/reasoning_val.json \
--output_path ./output/qwenvl_chartqa.jsonl \
--image_dir_path /mnt/petrelfs/share_data/mmtool/CharXiv/images \
--dataset_name chartqa \
--inferencer_name qwen2vl \
--model_name Qwen2-VL-7B-Instruct \
--controller_addr $controller_addr \
--max_rounds 3 \
