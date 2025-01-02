import json
import random
import torch
import pickle
from io import BytesIO
from PIL import Image
from grounding_parser import parse_resize, box2txt, boxes2txt,point2txt

IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
OBJS_PLACEHOLDER = '<objs>'
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'

        
def process_fn_CommonWebDataset(args, vis_processor, text_processor, cross_image_processor,  src):
    for data in src:
        # vison
        try:
            img_bytes = data['png'] if 'png' in data else data['jpg']
        except:
            from sat.helpers import print_rank0
            print_rank0(data.keys())
            print_rank0(data['__key__'])
            raise BaseException("{},{},{},{}".format('NONONONO!!! ', data.keys(), data['__key__'], data['__url__']))
        try:
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        img_dict = {'vision': vis_processor(img)}
        if cross_image_processor:
            img_dict.update({'cross': cross_image_processor(img)})

        # texts
        conversations = pickle.loads(data['conversations.pyd'])
        assert isinstance(conversations, list)
        if 'target.json' in data:
            target = json.loads(data['target.json'])


        # format placeholders
        scale, new_width, new_height = parse_resize(img, 400, 14)
        for conv in conversations:
            try:
                if 'boxes_seq' in conv and conv['boxes_seq']:
                    for bs in conv['boxes_seq']:
                        # for b in bs:
                        #     box = target['boxes'][b]
                        #     boxtxt = box2txt({'box':box}, scale, new_width, new_height)
                        #     conv['value'] = conv['value'].replace(BOXES_PLACEHOLDER, boxtxt, 1)
                        boxes = [target['boxes'][b] for b in bs]
                        boxestxt = boxes2txt({'boxes': boxes}, scale, new_width, new_height)
                        conv['value'] = conv['value'].replace(BOXES_PLACEHOLDER, boxestxt, 1)
                if 'points_seq' in conv and conv['points_seq']:
                    for ps in conv['points_seq']:
                        # for p in ps:
                        #     point = target['points'][p]
                        #     pointtxt = point2txt([point], scale, new_width, new_height)
                        #     conv['value'] = conv['value'].replace(POINTS_PLACEHOLDER, pointtxt, 1)
                        points = [target['points'][p] for p in ps]
                        pointstxt = point2txt(points, scale, new_width, new_height)
                        conv['value'] = conv['value'].replace(POINTS_PLACEHOLDER, pointstxt, 1)
                # Remove image tag
                for img_tag in ['\n'+IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER+'\n']:
                    conv['value'] = conv['value'].replace(img_tag, '')
            except:
                from sat.helpers import print_rank0
                print_rank0("{}".format(conv))
                print_rank0(f"Error happened in instruct_s1_dataset with {data['__url__']}")
                continue

        history = []
        for i in range(0, len(conversations), 2):
            question = conversations[i]['value']
            answer = conversations[i+1]['value']
            ret = {}
            # prompt = text_processor.history_to_prompt(history, question, add_eoi_first=True)
            prompt = question
            text_dict = text_processor(answer, prompt)
            if text_dict is None:
                continue
            ret.update(text_dict)
            history.append((question, answer))
            # ret.update(img_dict)
            # yield ret
            
            img_dict_flat = {'vision_'+k: v for k,v in img_dict['vision'].items()}
            ret.update(img_dict_flat)
            result_turns = [ret] # multi-turn
            yield result_turns

from sat.data_utils.webds import SimpleDistributedWebDataset
from functools import partial

def InstructDatasetProcessor(urls, args, vis_processor, text_processor, cross_image_processor=None, **kwargs):
    return SimpleDistributedWebDataset(urls, partial(process_fn_CommonWebDataset, args, vis_processor, text_processor, cross_image_processor), args.seed)