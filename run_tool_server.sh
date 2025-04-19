source activate tool-server
openaiproxy_on; unset http_proxy; unset HTTP_PROXY
cd /mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tool_workers/scripts/launch_scripts
python start_server_config.py --config ./config/all_service_szc_new.yaml
