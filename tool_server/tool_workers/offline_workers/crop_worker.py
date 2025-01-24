import uuid
import os
import re
import io
import argparse

from PIL import Image
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt

logger = build_logger("crop_worker")

def generate(params):
    generate_param = params["param"]
    image = params["image"]
    image = load_image(image)
    
    ret = {"text": "", "error_code": 0}
    
    match = re.match(r'\[\s*([\d,\s]+)\s*\]', generate_param)
    if match:
        try:
            # Extract coordinates and point indices
            line_coords = list(map(int, match.group(1).split(',')))

            if len(line_coords) != 4:
                raise ValueError(f"Invalid number of coordinates: {line_coords}")

            image = image.crop(line_coords)
            image_base64= pil_to_base64(image)
            ret["edited_image"] = image_base64
            ret["text"] = f"Image cropped successfully."
        except ValueError as e:
            logger.error(f"Error processing line parameters '{generate_param}': {e}")
            ret["text"] = f"Error processing line parameters '{generate_param}': {e}"
    else:
        logger.error(f"Parameter format mismatch: {generate_param}")
        ret["text"] = f"Parameter format mismatch: {generate_param}"
    
    return ret