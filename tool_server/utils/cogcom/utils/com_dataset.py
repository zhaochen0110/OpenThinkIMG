"""answer question base on the content of the image
@File    :   com_dataset.py
@Time    :   2024/2/4
@Author  :   Ji Qi 
@Contact :   qj20@mails.tsinghua.edu.cn
"""
import json, re
import random
import torch
import torch.utils.data as data
import pickle
import collections
from io import BytesIO
from PIL import Image
import glob
from .grounding_parser import parse_resize, boxes2txt, unrefine_box
from sat.helpers import print_rank0
from num2words import num2words
from .com_utils import HARD_PROMPT, build_tree, find_paths, manipulate_image


# prompts to start COM
start_prompts = [
    "Given a question, please solve the question step-by-step with a chain of manipulations. {QUESTION}",
    "Given a question, please solve the question step-by-step with a chain of manipulations, where in each step you can optionally use one of the following manipulations (defined as Name(Input)->Return) on the image: GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or a new manipulation created by yourself (if it is indeed needed). {QUESTION}",
    "{QUESTION} Please proceed with the question by optionally applying a series of manipulations like GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or even invent a new manipulation if required.",
    "When presented with a problem, please tackle it through a series of manipulations. For any step, you can choose to utilize one of the following manipulations on the image: GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or you might need to create a new one if it is indeed needed. {QUESTION}",
    "Take on the problem step by step, at each step you have a choice to apply one of the manipulatios such as GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or devising a new manipulation if there's a need. {QUESTION}",
    "Please tackle a given question in a step-by-step manner. For each step one of the following manipulations (depicted as Name(Input)->Retrun) can be optionally used: GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or develop a new manipulation yourself (if it is indeed required). {QUESTION}",
    "{QUESTION} Answer tihs question step-by-step selectively using the following manipulations: GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or devise a new manipulation if required.",
    "Address the given question step by step, using the provided manipulations (if necessary) such as GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers. If necessary, consider creating a new manipulation. {QUESTION}",
    "Please solve the problem gradually via a chain of manipulations, where in each step you can selectively adopt one of the following manipulations GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or invent a new manipulation, if that seems helpful. {QUESTION}",
    "Please go through the question incrementally with chain of manipulations (optionally use manipulation when needed) such as GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, and create a new manipulation if necessary. {QUESTION}",
    "{QUESTION} Solve the question step by step optionally utilizing manipulations like GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or even inventing a new manipulation if required.",
    "Kindly break down the given question into a chain of manipulations, and there are existing manipulations can be optionally used in each step: GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or even generating a new manipulation, if it proves necessary. {QUESTION}",
    "Answer the question in a step-by-step method, choosing to use manipulations such as GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or creating a new manipulation if needed. {QUESTION}",
    "{QUESTION} Please dissect this problem step-by-step, seeking helps from manipulations such as GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers; if need arises, create other manipulations.",
    "Please solve this question in a chain of steps, where each step you could optionally involves a manipulation such as GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers, or possibly a new manipulation created by you, if necessary. {QUESTION}",
    "When presented a question, systematically solve it using a series of manipulations. Each step might involve the use of one of the available manipulations (portrayed as Name(Input)->Return) on the image: GROUNDING(a phrase)->boxes, OCR(an image or a region)->texts, CROP_AND_ZOOMIN(a region on given image)->new_image, CALCULATE(a computable target)->numbers. If required, do not hesitate to create a new manipulation. {QUESTION}",
]

crop_and_zoomin_prompts = [
    "Using CROP_AND_ZOOMIN({BBX}, {X}) to crop the current image along the rectangular box {BBX} and enlarge the cropped image by {X} times to get a new image, and then re-input the new image.",
    "Employ the manipulation CROP_AND_ZOOMIN({BBX}, {X}) on box {BBX} to trim the current image, thereafter, magnify the trimmed image by {X} times, get a new image, and re-input this new image.",
    "Apply CROP_AND_ZOOMIN({BBX}, {X}) to current image to crop the current image along {BBX} and zoom in the the cropped image by {X} times, which should then be fed back as input.",
    "Utilize the CROP_AND_ZOOMIN({BBX}, {X}) method to snip the present picture following the dimensions of the rectangular box {BBX}, then zoom in the resulting crop by {X} times the size to craft a new image, and re-enter it.",
    "Activate the CROP_AND_ZOOMIN({BBX}, {X}) manipulation to crop the image along the {BBX} rectangle, then expand it {X} times over, resulting in a new image which should then be re-input.",
    "Invoke CROP_AND_ZOOMIN({BBX}, {X}) to cut the current image around {BBX}, inflate the cropped fragment {X} times its size to create a new image, and then re-feed this image.",
    "Execute CROP_AND_ZOOMIN({BBX}, {X}) to trim the current image as per {BBX}, amplify it by {X} times to fabricate a new image, which then should be re-entered.",
    "By using CROP_AND_ZOOMIN({BBX}, {X}) to crop the image along the rectangular box and magnify it by {X} times, thereby creating a new image for re-input.",
    "Leverage CROP_AND_ZOOMIN({BBX}, {X}) method to trim the present image, enlarge its size by {X} times to produce a different image and subsequently reintroduce it.",
    "Apply the CROP_AND_ZOOMIN({BBX}, {X}) operation on {BBX} to crop the image around the same region, magnify the resultant image {X} times over to procure a new image, which is then re-inputted.",
    "Implement CROP_AND_ZOOMIN({BBX}, {X}) on the specific region {BBX}, crop the extant image, increase the cropped part by {X} times to construct a new image and then re-input this image.",
    "Take advantage of CROP_AND_ZOOMIN({BBX}, {X}) manipulation on region {BBX}, act to crop the current image accordingly, zoom in the cropped part by {X} times to forge a new image, and re-input this new image.",
    "By leveraging CROP_AND_ZOOMIN({BBX}, {X}) on {BBX}, crop the image within the said zone, enlarge the resulting crop by {X} times, resulting in a new image to be re-input.",
    "Use the CROP_AND_ZOOMIN({BBX}, {X}) manipulation on the area identified as {BBX}, crop the ongoing image along this area, enlarge the crop by {X} times to its size to produce a new image which should then be re-inputted.",
    "By harnessing the CROP_AND_ZOOMIN({BBX}, {X}) manipulation for {BBX}, crop the existing image accordingly, magnify the cropped portion {X} times to yield a new image, and re-input the new image.",
    "Engage CROP_AND_ZOOMIN({BBX}, {X}), crop the image around this rectangle and expand it by {X} times to spawn a new image, followed by a re-input.",
    "Run the CROP_AND_ZOOMIN({BBX}, {X}) manipulation on {BBX}, crop the image around the rectangular box, enlarge the resultant piece by {X} times to get a replacement image, and feed it back.",
    "Execute the CROP_AND_ZOOMIN({BBX}, {X}) manipulation on {BBX}, crop the identified image along this rectangle, magnify it by {X} times over to create a new image, and introduce it back as an input.",
    "Utilize the CROP_AND_ZOOMIN({BBX}, {X}) manipulation on {BBX} for cropping the existing picture, multiply by {X} the size of the cropped section, produce a new picture which is then re-inputted.",
    "Engage the CROP_AND_ZOOMIN({BBX}, {X}) on the {BBX} area, crop the existing image along this box, amplify it {X} times to yield the new image, which should then be reintroduced as input."
]

# prompts to start COM
end_prompts = [
    "So the answer is {ANSWER}.",
    "Therefore, the answer to the question is {ANSWER}.",
    "Hence the final answer is {ANSWER}.",
    "Consequently, the ultimate answer to the question is {ANSWER}.",
    "Thus, the concluding response to your question is {ANSWER}.",
    "In conclusion, the final answer to this question is {ANSWER}.",
    "As a result, the ultimate conclusion to the question is {ANSWER}.",
    "So ultimately, the conclusive answer to the question in discussion is {ANSWER}.",
]


class Node:
    def __init__(self, data, key):
        self.key = key
        self.data = data

def build_tree(dict_tree):
    dtree = {}
    max_depth = 0
    for k,v in dict_tree.items():
        k = k.replace('*', '~')
        dtree[k] = v
        max_depth  = (k.split('--')[1].split(',')[0])

    tree = {}
    for key in sorted(dtree.keys()):
        fid, curid = key.split('--')
        flevel, ford = fid.split(',')
        clevel, cord = curid.split(',')
        if ford == '~':
            for i in range(1000): # maximum
                fid = f'{flevel},{i}'
                if fid not in tree:
                    if fid.split(',')[0] == '-1' and i==0: # one root
                        tree[fid] = [Node(dtree[key], key)]
                        if clevel != max_depth:
                            tree[curid] = [] # add current node !
                    break
                tree[fid].append(Node(dtree[key], key))
                if clevel != max_depth:
                    tree[curid] = []
        else:
            tree[fid] = tree.get(fid, []) + [Node(dtree[key], key)]
            if clevel != max_depth:
                tree[curid] = []
    return tree


def find_paths(tree):
    paths = []
    level = 0
    path = []
    def dfs(node):
        nonlocal tree, path
        fid, curid = node.key.split('--')
        path.append(node)
        if 'found' in node.data and node.data['found']:
            # paths.append([n.data['found'] for n in path])
            paths.append([n.data for n in path])
        if curid in tree:
            for child in tree[curid]:
                dfs(child)
        path.pop()
    # for root in tree[0]:
    for root in tree.get('-1,0', []):
        dfs(root)
    return paths


def manipulate_image(pil_image, func, param, unrefine=True) -> Image:
    """ Manipulate original image and produce a new image if possible, otherwise return the original image.
      Args:
        @pil_image: an image instance of PIL.Image
        @func: name of the manipulation function, where currently supporting: [crop_and_zoomin, ]
        @para: input parameter of the manipulation function
      Return:
        an produced (or original) image instance of PIL.Image
    """
    new_img = pil_image
    try:
        if func and 'crop_and_zoomin' in func.lower():
            box = None
            if isinstance(param, list) and param and (isinstance(param[0], int) or isinstance(param[0], float)):
                box = param
            elif isinstance(param, list) and param and isinstance(param[0], list) and (isinstance(param[0][0], int) or isinstance(param[0][0], float)):
                box = param[0]
            if box:
                if unrefine:
                    box = unrefine_box(box, new_img.size[0], new_img.size[1])
                new_img = new_img.crop(box)
                new_img = new_img.resize((new_img.size[0]*2, new_img.size[1]*2), Image.Resampling.BICUBIC)
    except:
        new_img = pil_image
        print_rank0(f'Failed to manipulate image with func: {func}, param: {param}')
    # print(f'Manipulating image from original size of {pil_image.size} to new size {new_img.size} ...')
    return new_img

        
def process_fn_COMWebDataset(args, vis_processor, text_processor, cross_image_processor,  src):
    for data in src:

        img_0_bytes = data['png'] if 'png' in data else data['jpg']
        img_0 = Image.open(BytesIO(img_0_bytes)).convert('RGB')

        metadata = pickle.loads(data['metadata.pyd'])
        for ex in metadata:

            com_tree = build_tree(ex['final_com'])
            com_chains = find_paths(com_tree)

            # initial text
            init_prompt = random.choice(start_prompts).format(QUESTION=ex['question'])
            final_answer = random.choice(end_prompts).format(ANSWER=ex['answer'])
            
            com_founds = ex['com_founds']
            # for chain in ex['com_chains']:
            for chain in com_chains:
                
                # Currently, store complete-prompt at each turn, and new-image at each turn.
                # TO-DO: store new-prompt, and new-image at each turn, and use history+new when inputing to model at each turn
                result_turns = [] # multi-turn

                imgs_turns = []
                # img_0 = Image.open(ex['img_path']).convert('RGB')
                imgs_turns.append(img_0)

                txt_turns = [] # (prompt, answer)
                # txt_turns.append({'prompt': '<EOI>' + init_prompt, 'answer': ""})
                txt_turns.append({'prompt': init_prompt, 'answer': ""})

                turn = 0
                history = []
                history_accum = ""
                ch_variables = {}
                ii = 0
                max_turn = 3
                # cropped = False
                cropped = ex.get('cropped', False)
                # while ii < len(chain):
                while ii < len(chain) and turn<max_turn:
                # for ii, stp in enumerate(chain):
                    # func, param, variables, onbox, ret, desc, found = stp['func'], stp['param'], stp['variables'], stp['onbox'], stp['return'], stp['desc'], stp['found']
                    stp = chain[ii]
                    func, param, variables, onbox, ret, desc, found = stp['func'], stp['param'], stp['variables'], stp['onbox'], stp['return'], stp['desc'], stp['found']
                    if variables:
                        ch_variables.update(variables)

                    img = imgs_turns[turn]
                    # img_dict = {'vision': self.vis_processor(img)}
                    img_dict = {'vision_'+k: v for k,v in vis_processor(img).items()}
                    if cross_image_processor:
                        img_dict.update({'cross_'+k: v for k,v in cross_image_processor(img).items()})
                    scale, width, height = parse_resize(img, 400, 14)

                    # replace variables
                    # if variables:
                    if ch_variables:
                        # ptr = re.compile(r'`[a-z]+_\d+`')
                        ptr = re.compile(r'`[a-z\_\d]+`')
                        for var in ptr.findall(desc):
                            pure_var = var[1:-1]
                            # value = variables.get(pure_var, var)
                            value = ch_variables.get(pure_var, var)
                            value = value if value else var
                            if type(value) == list:
                                boxes = value if type(value[0])==list else [value]
                                value = boxes2txt({'boxes':boxes}, scale, width, height) # using CogVLM transformation
                            # elif 'img' in value:
                            #     # value = "the new image"
                            #     value = "the new image <%s>" % value[1:-1].upper()
                            # desc = desc.replace(var, value)
                            desc = re.sub(f'{var}', f'{value}', desc)
                            
                    # Insert crop_and_zoomin manipulation:
                    #    (1) 'onbox' is much smaller than original image; (2) enlarge X=2/3/4 times
                    if random.random() <= 1.0 and not cropped and isinstance(onbox, list) and onbox and (isinstance(onbox[0], int) or isinstance([onbox[0]], float)):
                        area_i = img_0.size[0] * img_0.size[1]
                        area_b = (onbox[2]-onbox[0]) * (onbox[3]-onbox[1])
                        zoom_x = 1
                        if area_i / area_b >= 100:
                            zoom_x = 4
                        elif area_i / area_b >= 64:
                            zoom_x = 3
                        elif area_i / area_b >= 36:
                            zoom_x = 2
                        if zoom_x > 1:
                            cz_prompt = random.choice(crop_and_zoomin_prompts)
                            scale0, w0, h0 = parse_resize(img, 400, 14)
                            cz_bx = boxes2txt({'boxes':[onbox]}, scale0, w0, h0) # using CogVLM transformation
                            # cz_prompt = cz_prompt.format(BBX=cz_bx, X=num2words(zoom_x, to='cardinal'))
                            cz_prompt = cz_prompt.format(BBX=cz_bx, X=zoom_x)
                            ret, func, desc = 'img', 'crop_and_zoomin', cz_prompt
                            cropped = True
                            ii -= 1

                    txt_turns[turn]['answer'] = str.strip(txt_turns[turn]['answer'] + ' ' + desc)
                    if ii == len(chain)-1: # last trun
                        txt_turns[turn]['answer'] = txt_turns[turn]['answer'] + ' ' + final_answer

                    if (ret and 'img' in ret) or ii==len(chain)-1: # Start a new turn || The last turn
                        # text
                        # prompt = self.text_processor.history_to_prompt(history=[], question=txt_turns[turn]['prompt'], add_eoi_first=True)
                        result = {}
                        # text_dict = text_processor(txt_turns[turn]['answer'], prompt=txt_turns[turn]['prompt'], history=history)
                        text_dict = text_processor(txt_turns[turn]['answer'], prompt=txt_turns[turn]['prompt'], history=history_accum)
                        if text_dict is None:
                            ii += 1
                            continue
                        result.update(text_dict)
                        # image
                        result.update(img_dict)
                        result_turns.append(result) # a finished turn
                        
                        if ret and 'img' in ret:
                            # new prompt
                            # new_prompt = txt_turns[turn]['answer'] + '<EOI>' # mark image position
                            # new_prompt = '<EOI>Based on this new image and the reasoning history:' # mark image position
                            new_prompt = HARD_PROMPT
                            txt_turns.append({'prompt': new_prompt, 'answer':''})

                            # Produce a new image
                            # if 'crop_and_zoomin' in func.lower():
                            #     new_img = img.crop(onbox)
                            #     imgs_turns.append(new_img)
                            # else:
                            #     imgs_turns.append(img)
                            new_img = manipulate_image(img, func, onbox)
                            imgs_turns.append(new_img)
                        history.append((txt_turns[turn]['prompt'], txt_turns[turn]['answer']))
                        if history_accum:
                            history_accum = history_accum + ' ' + txt_turns[turn]['prompt'] + ' ' + txt_turns[turn]['answer']
                        else:
                            history_accum = txt_turns[turn]['prompt'] + ' ' + txt_turns[turn]['answer']
                        turn += 1
                    ii += 1
                if not result_turns or not result_turns[0]:
                    continue
                yield result_turns

from sat.data_utils.webds import SimpleDistributedWebDataset
from functools import partial

def CoMDatasetProcessor(urls, args, vis_processor, text_processor, cross_image_processor=None, **kwargs):
    return SimpleDistributedWebDataset(urls, partial(process_fn_COMWebDataset, args, vis_processor, text_processor, cross_image_processor), args.seed)