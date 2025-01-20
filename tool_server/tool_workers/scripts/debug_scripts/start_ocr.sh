cd /mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/tool_workers 

gpus=0
cpus=2
quotatype="reserved"

OMP_NUM_THREADS=1 srun --partition=MoE --job-name="zc_ocr" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} -w SH-IDCA1404-10-140-54-119 \
python ./ocr_worker.py \
--controller-address http://10.140.54.119:20001 

