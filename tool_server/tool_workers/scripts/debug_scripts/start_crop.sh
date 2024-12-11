source ~/.bashrc
source ~/anaconda3/bin/activate tool_server

cd /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tool_workers/restructure_worker

export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

gpus=0
cpus=2
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_dino" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./crop_worker.py \
--controller-address http://10.140.54.5:20001
