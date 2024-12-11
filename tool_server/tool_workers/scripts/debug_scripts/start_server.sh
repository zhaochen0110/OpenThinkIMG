# environment variables
source ~/.bashrc
source ~/anaconda3/bin/activate llava_plus

cd /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/serve

# export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

log_folder=/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/serve/logs/server_log

##PORT SETTINGS
# Controller: 20001
# Dino: 20003
# SAM: 20005
# LLAVA: 40000
# Gradio: 7860


## Start Controller
gpus=0
cpus=2
quotatype="auto"
log_file=${log_folder}/controller.log
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_controller" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 --output=${log_file} \
 python controller.py --host 0.0.0.0 --port 20001 &

# Step 2: 等待任务分配 Job ID
while true; do
    job_id=$(squeue --user=$USER --name="zc_controller" --noheader --format="%A" | head -n 1)
    node_list=$(squeue --job=$job_id --noheader --format="%N")
    if [ -n "$node_list" ]; then
        break
    fi
    sleep 1
done

node_list=$(squeue --job=$job_id --noheader --format="%N")
controller_addr=http://${node_list}:20001

echo "Controller address: ${controller_addr}"

## Start dino worker
gpus=1
cpus=16
quotatype="auto"
log_file=${log_folder}/dino_worker.log
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_dino" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 --output=${log_file} \
python ./grounding_dino_worker.py \
--host "0.0.0.0" \
--port 20003 \
--controller-address ${controller_addr} \
--model-config /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/dependencies/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
--model-path /mnt/petrelfs/songmingyang/songmingyang/model/tool-augment/groundingdino/groundingdino_swint_ogc.pth &

# Step 2: 等待任务分配 Job ID
while true; do
    job_id=$(squeue --user=$USER --name="zc_dino" --noheader --format="%A" | head -n 1)
    if [ -n "$job_id" ]; then
        break
    fi
    sleep 1
done




## Start Model worker
model_path=/mnt/petrelfs/songmingyang/songmingyang/model/tool-augment/llava_plus_v0_7b
gpus=1
cpus=16
quotatype="auto"
log_file=${log_folder}/llava_plus_worker.log
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_llava_plus_worker" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 --output=${log_file} \
 python -m llava_plus.serve.model_worker \
 --host "0.0.0.0" \
 --controller-address ${controller_addr} \
 --port 40000 \
 --worker-address auto \
 --model-path ${model_path} &

 ## Start sam worker
 gpus=1
 cpus=16
 quotatype="auto"
 log_file=${log_folder}/sam.log
 OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_sam" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 --output=${log_file} \
 python ./grounded_sam_worker.py \
 --controller-address ${controller_addr} \
 --port 20005 &


## Start a gradio worker
# gpus=0
# cpus=2
# quotatype="auto"
# log_file=${log_folder}/gradio.log
# OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_gradio" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
#  --output=${log_file} \
#  python -m llava_plus.serve.gradio_web_server_llava_plus \
#  --controller-url ${controller_addr} \
#  --model-list-mode reload &




