
source activate tool-server
openaiproxy_on; unset http_proxy; unset HTTP_PROXY
cd tool_workers
python start_server_config.py --config launch_tool.yaml
