source ~/.bashrc
source activate tool-server
# openaiproxy_on

rm -rf /mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/processed_data

config_path=/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/test/test_charxiv_qwen.yaml 
accelerate_path=/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/accelerate_config/cpu.yaml
model_args=model_name=gemini-2.0-flash-exp,max_retry=5,temperature=0.6
# model_args="pretrained=/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-7B-Instruct"
# model_name="qwen2vl"
model_name="gemini"

rm -rf ./${model_name}_caogao.jsonl

accelerate launch  --config_file ${accelerate_path} \
-m tool_server.tf_eval \
--model ${model_name} \
--model_args ${model_args} \
--task_name charxiv \
--model_mode "general" \
--verbosity INFO \
--output_path ./${model_name}_caogao.jsonl \
--batch_size 1 \
--max_rounds 8 

python read.py


# accelerate launch  --config_file  /mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/accelerate_config/cpu.yaml \
# -m tool_server.tf_eval \
# --config {$config_path}