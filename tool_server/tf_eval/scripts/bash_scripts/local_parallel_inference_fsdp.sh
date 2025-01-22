source ~/.bashrc
source ~/anaconda3/bin/activate tool_server

# environment variables
export OMP_NUM_THREADS=4
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server
cd $code_base
job_id=4034906
export SLURM_JOB_ID=${job_id}


export API_TYPE=openai
export OPENAI_API_URL=https://api.datapipe.app/v1/chat/completions
export OPENAI_API_KEY=sk-B3bRcR0fLubdoSmJ2cE13e57708c439aA14f825eB5Eb25De

accelerate_config=/mnt/petrelfs/songmingyang/.config/accelerate/4gpus.yaml
accelerate_config=/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tf_eval/scripts/configs/accelerate_config/4gpus_deepspeed.yaml
config_file=$1


gpus=4
cpus=32
quotatype="reserved"
OMP_NUM_THREADS=4 srun --partition=MoE --jobid=${job_id} --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
accelerate launch  --config_file  ${accelerate_config} \
-m tool_server.tf_eval --config  ${config_file} 


# salloc --partition=MoE --job-name="eval" --gres=gpu:8 -n1 --ntasks-per-node=1 -c 64 --quotatype="reserved"
# salloc --partition=MoE --job-name="interact" --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --quotatype="reserved"
