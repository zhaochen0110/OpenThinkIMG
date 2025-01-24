source ~/.bashrc
source activate tool-server

config_path=/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/test/test_charxiv_qwen.yaml 
accelerate_path=/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/accelerate_config/1gpu.yaml
model_args=pretrained=/mnt/petrelfs/share_data/mmtool/weights/qwen-cogcom-filter
# model_args="pretrained=/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-7B-Instruct"
# model_name="qwen2vl"
model_name="qwen2vl"
accelerate launch  --config_file ${accelerate_path} \
-m tool_server.tf_eval \
--model ${model_name} \
--model_args ${model_args} \
--model_mode "llava_plus" \
--task_name charxiv \
--verbosity INFO \
--output_path ./tool_server/tf_eval/scripts/logs/results/charxiv/${model_name}_cogcom.jsonl \
--batch_size 1 \
--max_rounds 3 \


# accelerate launch  --config_file  /mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/accelerate_config/cpu.yaml \
# -m tool_server.tf_eval \
# --config {$config_path}