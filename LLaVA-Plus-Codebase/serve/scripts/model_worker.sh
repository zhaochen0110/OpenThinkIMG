source ~/.bashrc
source ~/anaconda3/bin/activate llava_plus
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

cd /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/serve

export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

model_path=/mnt/petrelfs/songmingyang/songmingyang/model/tool-augment/llava_plus_v0_7b


gpus=1
cpus=16
quotatype="auto"
OMP_NUM_THREADS=8 srun --partition=MoE --job-name="zc_llava_plus" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
 python -m llava.serve.model_worker \
 --host 0.0.0.0 \
 --controller-address http://SH-IDCA1404-10-140-54-5:20001 \
 --port 40000 \
 --worker-address auto \
 --model-path ${model_path}
