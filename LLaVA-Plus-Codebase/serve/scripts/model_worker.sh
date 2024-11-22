source ~/.bashrc
source ~/anaconda3/bin/activate llava_plus

cd /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/serve

export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

gpus=1
cpus=16
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_controller" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 -w SH-IDCA1404-10-140-54-124 \
 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:20001 --port 40000 --worker http://localhost:40000 --model-path <huggingface or local path>
 