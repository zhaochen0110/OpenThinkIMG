source ~/.bashrc
source ~/anaconda3/bin/activate llava_plus

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

code_base=/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/dependencies/Grounded-Segment-Anything/GroundingDINO
cd $code_base
# job_id=3786651

export SLURM_JOB_ID=${job_id}
unset SLURM_JOB_ID
# config_file=$1

export API_TYPE=openai
export OPENAI_API_URL=https://api.datapipe.app/v1/chat/completions
export OPENAI_API_KEY=sk-0zMtaFA6orOE29d042561d2b9bE5407bA52151605b28Ea54

gpus=1
cpus=16
quotatype="reserved"
OMP_NUM_THREADS=1 srun --partition=MoE --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
pip install -e . --verbose > install_log.txt 2>&1


# salloc --partition=MoE --job-name="eval" --gres=gpu:8 -n1 --ntasks-per-node=1 -c 64 --quotatype="reserved"
# salloc --partition=MoE --job-name="interact" --gres=gpu:1 -n1 --ntasks-per-node=1 -c 16 --quotatype="reserved"