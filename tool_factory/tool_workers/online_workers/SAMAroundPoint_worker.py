"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import argparse
import torch
import numpy as np
from PIL import Image
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

import numpy as np

np.random.seed(3)


def extract_points(generate_param, image_w, image_h):
    all_points = []
    pattern = r'x\d*=\s*\\?"?([0-9]+(?:\.[0-9]*)?)\\?"?\s*y\d*=\s*\\?"?([0-9]+(?:\.[0-9]*)?)\\?"?'
    
    for match in re.finditer(pattern, generate_param):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            continue
        else:
            point = np.array(point)
            if np.max(point) > 100:
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    
    return all_points

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    fig, ax = plt.subplots()
    ax.imshow(image)
    image_format = image.format.lower() if image.format else 'png'
    if image_format not in ['png', 'jpeg', 'jpg']:
        image_format = 'png'
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, ax, borders=borders)
        if len(scores) > 1:
            ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    
    if point_coords is not None and input_labels is not None:
        show_points(point_coords, input_labels, ax)
    
    if box_coords is not None:
        show_box(box_coords, ax)
    
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    edited_image = Image.open(buf).convert("RGB")
    
    return edited_image

class SAM2ToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_name = "SegmentRegionAroundPoint",
                 model_path = "", 
                 model_base = "", 
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 sam2_checkpoint = "/mnt/petrelfs/share_data/suzhaochen/models/sam2-hiera-large/sam2_hiera_large.pt",
                 sam2_model_cfg = "sam2_hiera_l.yaml",
                 ):
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_cfg = sam2_model_cfg
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
        logger.info(f"Initializing model {self.model_name}...")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        sam2_checkpoint = self.sam2_checkpoint
        model_cfg = self.sam2_model_cfg

        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        self.predictor = SAM2ImagePredictor(self.sam2_model)

        
    def generate(self, params):
        generate_param = params["param"]
        image = params["image"]
        
        if generate_param is None or image is None:
            logger.error("Missing 'param' or 'image' in the input parameters.")
            return {"text": "Missing 'param' or 'image' in the input parameters.", "edited_image": None}
        
        ret = {"text": "", "error_code": 0}
        
        try:
            image = base64_to_pil(image)  

            img = image.convert("RGB")
            width, height = img.size
            self.predictor.set_image(img)

            points = extract_points(generate_param, width, height)

            if not points:
                logger.error("No valid points extracted.")
                ret["text"] = "No valid points extracted."
                ret['edited_image'] = None
                return ret

            input_labels = np.ones(len(points))
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=input_labels,
                box=None,
                multimask_output=False,
            )

            edited_img = show_masks(img, masks, scores, point_coords=np.array(points), input_labels=input_labels)
            ret['text'] = "Segmentation completed."
            ret['edited_image'] = pil_to_base64(edited_img)
            
        except Exception as e:
            logger.error(f"Error when using sam to SAMAroundPoint: {e}")
            ret["text"] = f"Error when using sam to SAMAroundPoint: {e}"
            ret['edited_image'] = None

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20036)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://SH-IDCA1404-10-140-54-119:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--sam2_checkpoint", type=str, default="/mnt/petrelfs/haoyunzhuo/mmtool/checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", type=str, default="sam2_hiera_l.yaml")
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = SAM2ToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg
    )
    worker.run()