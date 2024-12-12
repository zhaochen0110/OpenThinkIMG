# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate llava_plus


export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

gpus=0
cpus=2
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_controller" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 python controller.py --host 0.0.0.0 --port 20001 
 
 
#  python -m llava_plus.serve.controller --host 0.0.0.0 --port 20001

# curl -X POST http://10.140.54.5:20001/get_worker_address -d  '{"model":"grounding_dino"}'

