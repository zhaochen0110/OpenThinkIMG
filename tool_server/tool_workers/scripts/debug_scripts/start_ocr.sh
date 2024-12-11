source ~/.bashrc
source ~/anaconda3/bin/activate tool_server

AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address
unset http_proxy
unset HTTP_PROXY
unset https_proxy
unset HTTPS_PROXY
cd /mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tool_workers

export SLURM_JOB_ID=3273170
unset SLURM_JOB_ID     

gpus=0
cpus=2
quotatype="auto"

OMP_NUM_THREADS=1 srun --partition=MoE --job-name="zc_ocr" --mpi=pmi2 --gres=gpu:${gpus}  -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
python ./ocr_worker.py \
--controller-address http://10.140.54.5:20001 

