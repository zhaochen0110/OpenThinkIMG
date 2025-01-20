"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import argparse

from PIL import Image
from tool_server.utils import build_logger, pretty_print_semaphore
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

class DrawlineToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_path = "python/drawline_0.1", 
                 model_base = "python/drawline_0.1", 
                 model_name = "line",
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "auto",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore
            )
        
    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None
        
    def generate(self, params):
        generate_param = params["param"]
        image = params["image"]
        image = load_image(image)
        
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")
        
        ret = {"text": "", "error_code": 0}
        
        match = re.match(r'\[\s*([\d,\s]+)\s*\],\s*(\d+),\s*(\d+)', generate_param)
        if match:
            try:
                # Extract coordinates and point indices
                line_coords = list(map(int, match.group(1).split(',')))
                a, b = int(match.group(2)), int(match.group(3))

                if len(line_coords) != 4:
                    raise ValueError(f"Invalid number of coordinates: {line_coords}")

                x1, y1, x2, y2 = line_coords
                points = {
                    1: (x1, y1),  # Top-left corner
                    2: (x2, y1),  # Top-right corner
                    3: (x2, y2),  # Bottom-right corner
                    4: (x1, y2)   # Bottom-left corner
                }

                if a not in points or b not in points:
                    raise ValueError(f"Invalid point indices: a={a}, b={b}")

                start, end = points[a], points[b]

                # Draw the line with default linewidth=2 and linestyle='dashed'
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color='#e6194b',
                    linewidth=2,
                    linestyle='dashed'
                )
                buf = io.BytesIO()
                fig.savefig(buf, format='JPEG', bbox_inches='tight', pad_inches=0)  
                plt.close(fig)  # 关闭图形以释放资源
                buf.seek(0)  # 重置缓冲区指针到起始位置
                pil_image = Image.open(buf).convert("RGB")
                image_base64= pil_to_base64(pil_image)
                ret["edited_image"] = image_base64
                ret["text"] = f"Line drawn successfully."
            except ValueError as e:
                logger.error(f"Error processing line parameters '{generate_param}': {e}")
                ret["text"] = f"Error processing line parameters '{generate_param}': {e}"
        else:
            logger.error(f"Parameter format mismatch: {generate_param}")
            ret["text"] = f"Parameter format mismatch: {generate_param}"
        
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20013)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = DrawlineToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
    )
    worker.run()