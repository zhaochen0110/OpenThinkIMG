source ~/.bashrc
source activate vllm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3

openaiproxy_on
unset http_proxy
unset HTTP_PROXY

config_path=/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_factory/chartgemma_config/test_chartgemma_qwen.yaml

python -m tool_server.tf_eval --config ${config_path}
