# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate llava_plus

cd /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/serve

export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

gpus=0
cpus=2
quotatype="auto"

OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_gradio" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 python -m llava_plus.serve.gradio_web_server_llava_plus --controller http://10.140.54.5:20001 --model-list-mode reload
#  python -m llava_plus.serve.controller --host 0.0.0.0 --port 20001

# curl -X POST "http://10.140.54.5:21001/register_worker" -H "Content-Type: application/json" -d '{
#     "worker_name": "worker1",
#     "check_heart_beat": true,
#     "worker_status": {
#             "model_names": self.model_names,
#             "speed": 1,
#             "queue_length": self.get_queue_length(),
#         }
# }'