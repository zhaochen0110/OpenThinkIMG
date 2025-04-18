import os
import requests


from ..offline_workers import offline_tool_workers, get_tool_generate_fn
from tool_server.utils.utils import load_json_file
from tool_server.utils.server_utils import build_logger
from contextlib import contextmanager

logger = build_logger("tool_manager")

class ToolManager(object):
    def __init__(self, controller_url_location=None):
        self.controller_url_location = controller_url_location
        self.init_offline_tools()
        self.init_online_tools(self.controller_url_location)
        self.init_online_tool_addr_dict()
        logger.info(f"ToolManager is initialized.")
        self.available_tools = self.available_offline_tools + self.available_online_tools
        self.headers = {"User-Agent": "LLaVA-Plus Client"}
        
    def init_online_tools(self, controller_url_location=None):

        self.available_online_tools = []
        if controller_url_location is None:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../online_workers/controller_addr/controller_addr.json"
            logger.info("controller_addr is None, using default from controller_addr_location")
        else:
            self.controller_addr_location = controller_url_location
            logger.info(f"controller_addr exsits, controller_url_location is {controller_url_location}")
        
        if os.path.exists(self.controller_addr_location):
            self.controller_addr = load_json_file(self.controller_addr_location)["controller_addr"]
        else:
            self.controller_addr = self.controller_addr_location

        with self.disable_proxy():
            if self.controller_addr is not None and isinstance(self.controller_addr,str):
                ret = requests.post(self.controller_addr + "/refresh_all_workers")
                if ret.status_code == 200:
                    ret = requests.post(self.controller_addr + "/list_models")
                    models = ret.json()["models"]
                    logger.info(f"Online Tools: {models}")
                    self.available_online_tools = models
            

        miss_tool = []
        for tool in ["ZoomInSubfigure","DrawHorizontalLineByY","OCR","DrawVerticalLineByX","SegmentRegionAroundPoint","Point"]:
            if tool not in self.available_online_tools:
                miss_tool.append(tool)
        if len(miss_tool) == 0:
            logger.info("All tools are called successfully")
        else:
            logger.info(f"Not all tools is called successfully, missing tool {miss_tool}")        
    
    def init_offline_tools(self,):
        self.available_offline_tools = list(offline_tool_workers.keys())
        logger.info(f"Offline Tools: {self.available_offline_tools}")
        
    def init_online_tool_addr_dict(self):
        self.online_tool_addr_dict = {}
        for model_name in self.available_online_tools:
            with self.disable_proxy():
                ret = requests.post(self.controller_addr + "/get_worker_address",
                                    json={"model": model_name})
            worker_addr = ret.json()["address"]
            if worker_addr == "":
                logger.error(f"worker_addr for {model_name} is empty")
                continue
            self.online_tool_addr_dict[model_name] = worker_addr
    
    def call_tool(self,tool_name,params):
        if tool_name in self.available_offline_tools:
            try:
                tool_generate_fn = get_tool_generate_fn(tool_name)
                if tool_generate_fn is None:
                    return {"text": f"Tool {tool_name} not found.", "error_code": 1}
                else:
                    return tool_generate_fn(params)
            except Exception as e:
                logger.error(f"Failed to call tool {tool_name}: {e}")
                return {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}
            
        elif tool_name in self.available_online_tools:
            try:
                tool_worker_addr = self.online_tool_addr_dict[tool_name]
                with self.disable_proxy():
                    ret = requests.post(tool_worker_addr + "/worker_generate",headers=self.headers,json=params)
                return ret.json()
            except Exception as e:
                logger.error(f"Failed to call tool {tool_name}: {e}")
                return {"text": f"Failed to call tool {tool_name}: {e}", "error_code": 1}
        else:
            return {"text": f"Tool {tool_name} not found.", "error_code": 1}
            
    @contextmanager
    def disable_proxy(self):
        # 保存代理环境变量
        old_HTTP = os.environ.get("HTTP_PROXY")
        old_HTTPS = os.environ.get("HTTPS_PROXY")
        old_http = os.environ.get("http_proxy")
        old_https = os.environ.get("https_proxy")
        
        # 移除代理设置
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        
        try:
            yield
        finally:
            # 恢复代理设置
            if old_http is not None:
                os.environ["http_proxy"] = old_http
            if old_https is not None:
                os.environ["https_proxy"] = old_https
            if old_HTTP is not None:
                os.environ["HTTP_PROXY"] = old_HTTP
            if old_HTTPS is not None:
                os.environ["HTTPS_PROXY"] = old_HTTPS
