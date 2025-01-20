import uuid
import os
import re
import io
import argparse

from PIL import Image
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt

logger = build_logger("drawline_worker")

def generate(params):
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