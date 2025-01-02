cd /mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/tool_workers/restructure_worker    

gpus=0
cpus=2
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="crop" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./crop_worker.py \
--controller-address http://10.140.54.5:20001
