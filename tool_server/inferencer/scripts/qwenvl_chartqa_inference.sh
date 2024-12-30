controller_addr=http://SH-IDCA1404-10-140-54-89:20001

code_base=/mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/inferencer
cd $code_base

gpus=0
cpus=2
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="inference" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} -w SH-IDCA1404-10-140-54-89 \
nohup python ./singleloop_inference.py \
--input_path /mnt/petrelfs/share_data/mmtool/CharXiv/code-CharXiv/data/reasoning_val.json \
--output_path ./output/qwenvl_charxiv_10turns.jsonl \
--image_dir_path /mnt/petrelfs/share_data/mmtool/CharXiv/images \
--dataset_name chartqa \
--inferencer_name qwen2vl \
--model_name Qwen2-VL-7B-Instruct \
--controller_addr $controller_addr \
--max_rounds 10 1>/mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/inferencer/logs/Qwen_charxiv_cogcom_10turns.log 2>&1 &
