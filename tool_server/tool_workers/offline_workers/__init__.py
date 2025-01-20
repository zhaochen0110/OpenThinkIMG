
offline_tool_workers = {
    "crop":"crop_worker",
    "drawline":"drawline_worker",
}

def get_tool_generate_fn(tool_name):
    if tool_name not in offline_tool_workers:
        return None
    module = __import__(f"tool_server.tool_workers.offline_workers.{offline_tool_workers[tool_name]}", fromlist=["generate"])
    return getattr(module, "generate")