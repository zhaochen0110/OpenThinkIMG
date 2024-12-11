# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate llava_plus

cd /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/serve

export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

gpus=1
cpus=16
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_dino" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./grounding_dino_worker.py \
--controller-address http://10.140.54.5:20001


# srun --partition=MoE --mpi=pmi2 --job-name=pruner --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --quotatype=reserved \
#  python ../serve/grounding_dino_worker.py
# -w SH-IDCA1404-10-140-54-67 \