cd /mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/tool_workers/restructure_worker


OMP_NUM_THREADS=8 srun \
 --partition=MoE \
 --mpi=pmi2 \
 --job-name=molmo-p \
 -c 32 \
 -w SH-IDCA1404-10-140-54-89 \
 --ntasks-per-node=1 \
 --kill-on-bad-exit=1 \
 --quotatype=reserved \
 python ./molmo_point_worker.py 