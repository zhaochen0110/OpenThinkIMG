cd /mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/tool_workers/restructure_worker/test_cases   

gpus=0
cpus=2
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="test-sam" --mpi=pmi2 --gres=gpu:${gpus}  -w SH-IDCA1404-10-140-54-119  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ocr_test_message.py --send_image \