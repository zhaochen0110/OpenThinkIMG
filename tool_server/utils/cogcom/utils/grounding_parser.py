import math
import spacy
import re
import io
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import random

nlp = spacy.load("en_core_web_sm")

def unrefine_box(box, width, height, rounded=False):
    box = [box[0]/1000*width, box[1]/1000*height, box[2]/1000*width, box[3]/1000*height]
    if rounded:
        box = [round(b) for b in box]
    return box

def refine_box(box, scale, new_width, new_height):
    box = [min(box[0], new_width-1), min(box[1], new_height-1), min(box[2], new_width-1), min(box[3], new_height-1)]
    box = [box[0]/new_width, box[1]/new_height, box[2]/new_width, box[3]/new_height]
    box = [math.floor(x*1000) for x in box]
    if box[0] >= 1000 or box[1] >= 1000 or box[2] >= 1000 or box[3] >= 1000:
        print(str(box))
        box = [min(box[0], 999), min(box[1], 999), min(box[2], 999), min(box[3], 999)]
    return box

def get_text_by_box(boxlist, sep=" "):
    strs = [f"{box[0]:03d},{box[1]:03d},{box[2]:03d},{box[3]:03d}" for box in boxlist]
    random.shuffle(strs)
    return "{}[[{}]]".format(sep, ";".join(strs))

def box2txt(data, scale, new_width, new_height):
    box = refine_box(data['box'], scale, new_width, new_height)
    return get_text_by_box([box], sep="")

def boxes2txt(data, scale, new_width, new_height):
    ret = []
    for box in data['boxes']:
        ret.append(refine_box(box, scale, new_width, new_height))
    return get_text_by_box(ret, sep="")

def point2txt(point_list, scale, new_width, new_height, sep=" "):
    ret_points = []
    for point in point_list:
        point = [min(round(point[0]*scale), new_width-1), min(round(point[1]*scale), new_height-1)]
        point = [point[0]/new_width, point[1]/new_height]
        point = [math.floor(x*1000) for x in point]
        if point[0] >= 1000 or point[1] >= 1000:
            print(str(point))
            point = [min(point[0], 999), min(point[1], 999)]
        ret_points.append(point)
    strs = [f"{point[0]:03d},{point[1]:03d}" for point in ret_points]
    # random.shuffle(strs)
    return "{}[[{}]]".format(sep, ";".join(strs))

def parse_resize(img, h, w):
    return -1, img.size[0], img.size[1]
    # if type(img_bytes) is not Image.Image:
    #     img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # else:
    #     img = img_bytes.convert('RGB')
    totalpatch, lpatch = h, w
    # maximize scale s.t.
    scale = math.sqrt(totalpatch * (lpatch / img.size[1]) * (lpatch / img.size[0]))
    num_feasible_rows = max(min(math.floor(scale * img.size[1] / lpatch), totalpatch), 1)
    num_feasible_cols = max(min(math.floor(scale * img.size[0] / lpatch), totalpatch), 1)
    target_height = max(num_feasible_rows * lpatch, 1)
    target_width = max(num_feasible_cols * lpatch, 1)
    # img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
    # img = img.crop((0, 0, target_width, target_height))
    return scale, target_width, target_height


def vis_prop(img, boxes, new_width, new_height, file_name='output.png'):
    # new_img = img
    new_img = ImageDraw.Draw(img)
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    for boxs in boxes:
        for box in boxs:
            box = [box[0]/1000*new_width, box[1]/1000*new_height, box[2]/1000*new_width, box[3]/1000*new_height]
            # Create a Rectangle patch
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            # new_img = ImageDraw.Draw(new_img)
            new_img.rectangle(box, fill=None, outline='red')
    new_img = new_img._image
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(round(x/new_width, 3), ',')))
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(round(x/new_height, 3), ',')))
    if file_name:
        plt.savefig(file_name)
    plt.close()
    return new_img



def visualize(img, boxes, scale=14, file_name='output.png'):
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    for boxs in boxes:
        for box in boxs:
            box = [x*scale for x in box]
            # Create a Rectangle patch
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(round(x/scale, 1), ',')))
    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(round(x/scale, 1), ',')))
    plt.savefig(file_name, dpi=1200)
    plt.close()


def process(img, processor):
    num_patches = processor.args[0]
    # img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    scale = math.sqrt(num_patches * (14 / img.size[1]) * (14 / img.size[0]))
    new_img = parse_resize(img, num_patches, 14, 'patch-crop-2')
    inputs = processor(img)
    return new_img, inputs['image_size'], scale


def parse_response(img, response, img_processor, output_name='output.png'):
    if type(img_processor.args[0]) is int:
        new_img, new_size, _ = process(img, img_processor) # new_size: (h, w)
    else:
        img = img.convert('RGB')
        # width, height = img.size
        # ratio = min(1920 / width, 1080 / height)
        # new_width = int(width * ratio)
        # new_height = int(height * ratio)
        # new_img = img.resize((new_width, new_height), Image.LANCZOS)
        # new_img = img.resize((224, 224))
        new_img = img.resize((490, 490))
        new_size = new_img.size
    pattern = r"\[\[(.*?)\]\]"
    positions = re.findall(pattern, response)
    boxes = [[[int(y) for y in x.split(',')] for x in pos.split(';') if x.replace(',', '').isdigit()] for pos in positions]
    # visualize(new_img, boxes, scale=14)
    if type(img_processor.args[0]) is int:
        drawn_img = vis_prop(new_img, boxes, new_size[1]*14, new_size[0]*14, file_name='output.png')
    else:
        # vis_prop(new_img, boxes, 224, 224, file_name='output.png')
        # vis_prop(new_img, boxes, 224, 224, file_name=output_name)
        drawn_img = vis_prop(new_img, boxes, 490, 490, file_name=output_name)
    # visualize(img, [], scale=1, file_name='output_origin.png')
    return drawn_img
