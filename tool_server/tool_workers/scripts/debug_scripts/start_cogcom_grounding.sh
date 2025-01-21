cd /mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/tool_workers/online_workers


OMP_NUM_THREADS=8 srun \
 --partition=MoE \
 --mpi=pmi2 \
 --job-name=subplot \
 -c 32 \
 -w SH-IDCA1404-10-140-54-119 \
 --ntasks-per-node=1 \
 --kill-on-bad-exit=1 \
 --quotatype=reserved \
 python ./zoomInsubfigure_worker.py