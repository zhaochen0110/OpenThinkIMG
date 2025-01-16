source ~/.bashrc
source activate tool-server

config_path=/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/test/test_charxiv_qwen.yaml 
accelerate_path=/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/accelerate_config/cpu.yaml

accelerate launch  --config_file ${accelerate_path} \
-m tool_server.tf_eval \
--model gemini \
--model_args model_name=gemini-2.0-flash-exp,max_retry=5 \
--task_name charxiv \
--verbosity INFO \
--output_path ./tool_server/tf_eval/scripts/logs/results/charxiv/gemini.jsonl \
--batch_size 1 \
--max_rounds 3 \


# accelerate launch  --config_file  /mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tf_eval/scripts/configs/accelerate_config/cpu.yaml \
# -m tool_server.tf_eval \
# --config {$config_path}