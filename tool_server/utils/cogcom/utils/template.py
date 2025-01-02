# from email.mime import image
import string
import random
import pdb
import json
random.seed(7)


TEMPLATES = {
    "concise_post": [
        "Please give me a straightforward answer.",
        "I require a brief and clear answer for this question.",
        "Please take a look at the image and promptly provide an answer.",
        "Please provide me a concise answer based on the image",
        "Refer to the image and give a clear response.",
        "Please quickly answer this question in a simple manner.",
        "Please provide me a clear and direct answer to the question by analyzing the image",
        "Please give me an exact answer for this question by referring to the image.",
        "Please check the image and provide a straightforward answer for my question.",
        "Please keep your response brief and clear.",
        "I want a simple and direct answer to my question.",
        "Please examine the image and provide a straightforward answer to my question.",
        "Please take a look at the image and give a concise response to the question.",
        "Please you provide a brief and direct answer based on the image.",
        "I need a clear answer to this question in regards to the image.",
        "Give me a concise answer while keeping the image in mind.",
        "Answer this question directly after referring to the image.",
        "Please answer the following question: <question> with reference to the attached image <image>, in a clear and simple way.",
        "Please give me a direct response to my question.",
        "Please give me a simple and direct answer.",
        "Please be brief and clear.",
        "Please derive the answer to my question by observing the image, and keep it concise.",
        "Please reply with a straightforward answer after checking the image.",
        "Kindly provide me with a brief and clear answer to my question based on the image.",
        "Offer me a concise response to this query after analyzing the image.",
        "Relying on the image, please give me a direct answer to the question.",
        "Please give a simple answer without elaborating.",
        "Please give a concise and clear response while referring to the image.",
        "I'm seeking a straightforward answer based on the provided image",
        "Check the image and swiftly give me a brief response to my question.",
        "Please give me a straightforward answer by taking the image into consideration.",
        "Please provide a brief answer without any explanation?",
        "Upon examining the image, kindly provide a brief and clear response.",
        "A simple and direct answer to would be appreciated.",
        "Kindly give me a brief and to-the-point answer using the image as a reference.",
        "Please provide a succinct and precise response, taking into account the image.",
        "Give me a clear-cut answer in relation to the image.",
        "Answer in the most concise manner while using the image as a reference.",
        "Please give a straight-to-the-point response after considering the image.",
        "Refer to the image and provide a short and clear answer.",
        "In reference to the image, please answer without any unnecessary details.",
        "Taking into consideration the image, please respond concisely.",
        "Please provide a direct and to-the-point response while considering the image.",
        "Review the image and give me a short and precise answer.",
        "Please give me a to-the-point and focused response, after considering the image.",
        "With reference to the image, please answer in a clear and concise manner.",
        "In the context of the image, please provide a concise and direct answer.",
        "Give a simple and straight answer based on the information in the image.",
        "Please take a look at the image and provide a direct and crisp answer.",
        "In the context of the image, kindly provide a simple and to-the-point response.",
        "Offer a short and simple answer after observing the image.",
        "After looking at the image, please respond in a succinct manner.",
        "Please give a direct and focused reply while taking the image into account.",
        "Taking into account the image, provide a concise answer.",
        "I'm interested in a concise answer while taking the image into consideration.",
        "Please provide a straightforward and compact response while keeing the image in mind."
    ]
}



def build_instruction(task, text=None, options=None, region=None, context=None, question=None, explanation=None, response=None, premise=None, hypothesis=None, answer=None, meta_data=None, target=None, use_natural=False, instruction_id=-1):
    # if options:
    #     random.shuffle(options)
    image_token = "image" # this show only appear before the output token
    # options_token = "\n\n[Options]:"
    # region_token = "Regions: "
    options_tokens = ["\n\n[Options]:", "\n\nOptions:", "\n\nOptions:", "\n\noptions:"]
    region_tokens = ["Regions: ", "Bounding Boxes: "]
    options_token = random.choice(options_tokens)
    region_token = random.choice(region_tokens)

    # split_token = "||||" # or 
    # region_split_token = "||||" # or 
    split_tokens = ['|', '||', '||||', ';', ',']
    region_split_tokens = ['|', '||', '||||', ';', ',']
    split_token = random.choice(split_tokens)
    region_split_token = random.choice(region_split_tokens)

    concise_post = random.choice(TEMPLATES['concise_post'])
    
    if options:
        random.shuffle(options)
        # define options string
        if use_natural  == 'use_natural':
            num_choices = list(range(1, len(options)+1)) # 1, 2, 3, ..
            num_choices_b = [f'({c})' for c in num_choices] # (1), (2), (3), ..
            lower_letter_choices = [f'({c})' for c in string.ascii_lowercase] # (a), (b), (c), ..
            upper_letter_choices = [f'({c})' for c in string.ascii_uppercase] # (A), (B), (C), ..
            op_choices = [f'Option {c}:' for c in num_choices] # Option 1:, Option 2:, ...
            choices_list = [('', num_choices, ', '), ('', num_choices_b, ', '), ('', lower_letter_choices, ', '), ('', upper_letter_choices, ', '), ('\n', op_choices, '\n')]
            # tgt_choice = choices_list[instruction_id]
            tgt_choice = choices_list[random.randint(len(choices_list)-1)]
            
            options_str = [f'{tgt_choice[1][i]}. {option}' for i, option in enumerate(options)]
            options_str = tgt_choice[2].join(options_str)
            options_str = f'{options_token}{tgt_choice[0]} {options_str}'
        else:
            options_str = f'{options_token} {split_token.join(options)}'
    
    # --------------------------- training tasks ---------------------------
    if task == 'image_caption':
        instructs=[
            # f"""In this task, you will look at the image and briefly describe the image.""",
            f"""In this task, you will look at the image and describe the image. {concise_post}""",
            f"""What is the caption of the image? {concise_post}""",
            f"""Generate some text to describe the image. {concise_post}""",
            f"""Look at image and tell me what is the content. {concise_post}""",
            f"""In this task, you are given an image and you will need to generate some text to describe it. {concise_post}"""
        ]
    elif task == 'open-domain_VQA':
        instructs = [
            f"""{question}  {concise_post}""",
            f"""{question}  {concise_post}""",
            f"""{question}  {concise_post}""",
            f"""{question}  {concise_post}""",
            f"""{question} {concise_post}"""
        ]
    elif task == 'VQA':
        instructs = [
            f"{question}{options_token} {split_token. join(options)} {concise_post}",
            f"{question}{options_token} {split_token. join(options)} {concise_post}",
            f"{question}{options_token} {split_token. join(options)} {concise_post}",
            f"{question}{options_token} {split_token. join(options)} {concise_post}",
            f"{question}{options_token} {split_token. join(options)} {concise_post}"]
    elif task == 'GC':
        instructs = [
            f"""The goal of this task is to generate description for one part of the image. The part is specified by {region_split_token.join(region)}.""",
            f"""What is the content of {region_split_token.join(region)}?  {concise_post}""",
            f"""Describe the content of {region_split_token.join(region)} in image.  {concise_post}""",
            f"""Generate a caption for {region_split_token.join(region)}.  {concise_post}""",
            f"""{region_split_token.join(region)} is a region in image. Locate the region first and generate a description for that part of image.""",
        ]
    elif task == 'GC_selection':
        instructs = [
            f"""Select the description for one part of the image. The part is specified by {region_split_token.join(region)}.{options_token} {split_token.join(options)}  {concise_post}""",
            f"""What is the content of {region_split_token.join(region)}?{options_token} {split_token.join(options)} {concise_post}""",
            f"""Select the content of {region_split_token.join(region)} from options.{options_token} {split_token.join(options)} {concise_post}""",
            f"""What is the caption for {region_split_token.join(region)}?{options_token} {split_token.join(options)} {concise_post}""",
            f"""{region_split_token.join(region)} is a region in image. Select a description for that part of image.{options_token} {split_token.join(options)} {concise_post}""",
        ]
    elif task == 'VG':
        instructs = [
            # f"""The region in image that \"{text}\" describes is {concise_post}""",
            f"""The region in image that \"{text}\" describes is? {concise_post}""",
            f"""Find the region in image that \"{text}\" describes. {concise_post}""",
            f"""The goal of this task is to find the part of the image with the description: \"{text}\" {concise_post}""",
            f""" \"{text}\" describes part of the image. Find the part. {concise_post}""",
            f"""In this task, you are asked to localize the region in image that is described by the given text. The text is \"{text}\""""
        ]
    elif task == 'VG_selection':
        instructs = [
            f"""Select region in the image that \"{text}\" describes.{options_token} {split_token.join(options)} {concise_post}""",
            f"""What is the region in the image that \"{text}\" describes?{options_token} {split_token.join(options)} {concise_post}""",
            f"""The goal of this task is to select the region of the image with the description: \"{text}\"{options_token} {split_token.join(options)}""",
            f""" \"{text}\" describes part of the image. Find the part.{options_token} {split_token.join(options)} {concise_post}""",
            f"""In this task, you are asked to localize the region in image that is described by the given text. The text is \"{text}\"{options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'object_grounding':
        instructs = [
            f"""What is the object in {region_split_token.join(region)} {concise_post}""",
            f"""Identify the object in {region_split_token.join(region)}. {concise_post}""",
            f"""The goal of this task is to identify the object in given regions in image. The region is {region_split_token.join(region)}. What is the object? {concise_post}""",
            f"""The object contained in {region_split_token.join(region)} is what?  {concise_post}""",
            f"""In this task, you are given the coordinates of some rectangular region in the image. You need to first localize each rectangular region and then identify what is the object in the region. The region is {region_split_token.join(region)}."""
        ]
    elif task == 'object_region_match':
        instructs = [
            f"""Is the object \"{text}\" in {region_split_token.join(region)}? {options_token} {split_token.join(options)}  {concise_post}""",
            f"""Does the region {region_split_token.join(region)} contain \"{text}\"? {options_token} {split_token.join(options)} {concise_post}""",
            f"""Answer if the region {region_split_token.join(region)} contains \"{text}\". {options_token} {split_token.join(options)} {concise_post}""",
            f"""In this task, you will need to decide if the object in {region_split_token.join(region)} is \"{text}\". {options_token} {split_token.join(options)} {concise_post}""",
            f"""Decide if the object in {region_split_token.join(region)} matches \"{text}\". {options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'object_match':
        instructs = [
            f"""Are the object in {region[0]} and object in {region[1]} the same type? {options_token} {split_token.join(options)} {concise_post}""",
            f"""In this task you are given two objects. Each object is specified by its location in the image. One object is in {region[0]} and another object is in {region[1]}. Decide if two objects have the same type. {options_token} {split_token.join(options)}""",
            f"""The goal of this task is to check if two regions contain the same type of object in the image. The two regions are {region_split_token.join(region)}. {options_token} {split_token.join(options)} {concise_post}""",
            f"""Do objects in {region_split_token.join(region)} have the same type? {options_token} {split_token.join(options)} {concise_post}""",
            f"""Determine whether the same kind of object is present in both given regions of the image. The two regions are {region_split_token.join(region)}. {options_token} {split_token.join(options)} {concise_post}"""
        ]    
    elif task == 'question_image_match':
        instructs = [
            f"""In this task, you need to decide if the image has enough information to answer \"{question}\" {options_token} {split_token.join(options)} {concise_post}""",
            f"""Given content of image, do you have enough information to answer \"{question}\" {options_token} {split_token.join(options)} {concise_post}""",
            f"""In this task, you are given the question \"{question}\" and you need to decide if the image provide you enough info to answer the question. {options_token} {split_token.join(options)} {concise_post}""",
            f"""Is it possible to answer \"{question}\" given the content of image? {options_token} {split_token.join(options)} {concise_post}""",
            f"""Does the image contain the answer to \"{question}\"? {options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'object_region_selection':
        instructs = [
            f"""Select the region containing \"{text}\".{options_token} {split_token.join(options)} {concise_post}""",
            f"""What is the regions in the options that contain \"{text}\"?{options_token} {split_token.join(options)} {concise_post}""",
            f"""Which option contains \"{text}\"?{options_token} {split_token.join(options)} {concise_post}""",
            f"""Select the option that contains the object \"{text}\".{options_token} {split_token.join(options)} {concise_post}""",
            f"""You are given regions as options and select the option that contains the object \"{text}\".{options_token} {split_token.join(options)} {concise_post}""",
        ]
    # modify
    elif task == 'missing_object_selection':
        instructs = [f"""Select objects that do not appear in any of {region_split_token.join(region)}. Select "None" if you can't find any.{options_token} {split_token.join(options)} {concise_post}""",
                     f"""Select options that do not appear in any of {region_split_token.join(region)}.{options_token} {split_token.join(options)} {concise_post}""",
                     f"""Given {region_split_token.join(region)}, select objects that do not appear in any of the regions. Select "None" if you can't find it.{options_token} {split_token.join(options)} {concise_post}""",
                     f"""Which objects in options do not in appear in any of {region_split_token.join(region)}? Select "None" if you can't find it.{options_token} {split_token.join(options)} {concise_post}""",
                     f"""In this task, you are given some regions {region_split_token.join(region)}. Decide which object in options that do not appear in any of the given region.{options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'ITM':
        instructs = [f"""Does \"{text}\" describes image? {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Does the text: \"{text}\" and the content of image match? {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Is the text: \"{text}\" the caption of image? {options_token} {split_token.join(options)} {concise_post}""",
                     f"""In this task you are given some text and you need to decide if the text describe the image. {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Is the caption of image \"{text}\"? {options_token} {split_token.join(options)} {concise_post}""",
        ]
    # modify    
    elif task == 'region_object_selection': 
        instructs = [f"""Select objects from the options that appear in at least one of the regions. Select "None" if you can't find it.{region_token} {region_split_token.join(region)}. {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Given objects in the options, select options that appear in at least one of {region_split_token.join(region)}.Select "None" if you can't find any.{options_token} {split_token.join(options)} {concise_post}""",
                     f"""What are the objects in the options that appear in at least one of the regions: {region_split_token.join(region)}?{options_token} {split_token.join(options)} {concise_post}""",
                     f"""Given {region_token} {region_split_token.join(region)}, decide which object appears in at least one of the region.{options_token} {split_token.join(options)} {concise_post}""",
                     f"""Given some regions, select object that appears in at least one of the region. {region_token} {region_split_token.join(region)}{options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'region_generation': # mscoco
        instructs = [f"""What are the regions contain the object \"{text}\"? {concise_post}""",
                     f"""Given object: \"{text}\", what are the regions that contain this objects? {concise_post}""",
                     f"""The regions that contain \"{text}\" are what?  {concise_post}""",
                     f"""The parts of image that have \"{text}\" are what?  {concise_post}""",
                     f"""Identify the regions that contain \"{text}\".  {concise_post}""",
                     f"""In this task, you are asked to identify all the regions in the image that contain the object \"{text}\".  {concise_post}""",
                     f"""Which parts of image contain \"{text}\"? {concise_post}"""
                     ]
    elif task == 'region_caption_match':
        instructs = [f"""Decide if \"{text}\" is the description of {region_split_token.join(region)}. {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Does \"{text}\" matches the content of {region_split_token.join(region)}. {options_token} {split_token.join(options)} {concise_post}""",
                     f"""In this task, you need to decide if \"{text}\" is a caption of {region_split_token.join(region)} in the image. {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Can \"{text}\" describe {region_split_token.join(region)}? {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Does {region_split_token.join(region)} and given text match? Text: {text} {options_token} {split_token.join(options)} {concise_post}"""
                     ]
    elif task == 'object_relationship':
        instructs = [
            f"""In this task, you are given the regions of a subject A and an object B in the image. Determine what is their relationship. The relationship can be the position of subject A relative to object B or what is subject A doing to object B. Region of subject A is {region[0]} and region of object B is {region[1]}.""",
            f"""What is the relationship between the subject in {region[0]} and object in {region[1]}? {concise_post}""",
            f"""Given a subject in {region[0]} and an object in {region[1]}, what's their relationship? {concise_post}""",
            f"""Subject A: {region[0]} Object B: {region[1]} and their relationship is what?  {concise_post}""",
            f"""Tell me the relationship between the subject in {region[0]} and the object in {region[1]}. {concise_post}"""
        ]
    elif task == 'visual_object_identification':
        instructs = [
            f"""Given the image, the subject in {region[0]} {meta_data['relation']} what? {concise_post}""",
            f"""Given the image, the subject in {region[0]} {meta_data['relation']} an object. What is the object? {concise_post}""",
            f"""Given the subject in {region[0]} and relationship \"{meta_data['relation']}\". What is the object? {concise_post}""",
            f"""Identify the name of the object, given the subject in {region[0]} and relationship: {meta_data['relation']}.  {concise_post}""",
            f"""In this task, you are asked to identify the object given tne region of the subject in the image and their relationship. The subject is in {region[0]} and relationship is {meta_data['relation']}. The object is what?  {concise_post}""",
        ]
    elif task == 'visual_subject_identification':
        instructs = [
            f"""Given the image and the object in {region[1]}, predict what is the subject {meta_data['relation']} the object? {concise_post}""",
            f"""Given the object in {region[1]}, and the relationship {meta_data['relation']}. What is the subject. {concise_post}""",
            f"""Identify the subject that {meta_data['relation']} the object.\nThe object is in {region[1]}.  {concise_post}""",
            f"""Which subject in the image that has {meta_data['relation']} with the object in {region[1]}. {concise_post}""",
            f"""In this task, you are given the region of the object and the relation. What is the name of the subject? \n\nRelationship: {meta_data['relation']}\nObject: {region[1]}. {concise_post}""",
            
        ]
    elif task == 'visual_object_region':
        region=  region_split_token.join(meta_data['object_regions']['subject'])
        instructs = [
            f"""Which object has the relationship \"{meta_data['relation']}\" with the subject in {region}? Answer the question by generating the region of the object. {concise_post}""",
            f"""Find the region of the object that has the relationship \"{meta_data['relation']}\" with the subject in {region}. {concise_post}""",
            f"""Given the image, where is the object that has the relatipnship \"{meta_data['relation']}\" with the the subject in {region}? {concise_post}""",
            f"""Identify the region of the object given the subject in {region} and relationship \"{meta_data['relation']}\". {concise_post}""",
            f"""What is the object region, given subject in {region} and relationship \"{meta_data['relation']}\"? {concise_post}""",
            f"""What is the object region, given the subject region and the relationship?\n\nSubject region: {region} Relationship: \"{meta_data['relation']}\"? {concise_post}"""
        ]
    elif task == 'visual_subject_region':
        region=  region_split_token.join(meta_data['object_regions']['object'])
        instructs = [
            f"""Given the object in {region}, where is the subject in the image that has relationship: \"{meta_data['relation']}\" with the object? {concise_post}""",
            f"""The object is in {region}. Identify the region of the subject that has relationship: {meta_data['relation']} with the object. {concise_post}""",
            f"""What is the region of the object, given subject in {region} and relationship \"{meta_data['relation']}\"? {concise_post}""",
            f"""Subject is in {region} and relationship is \"{meta_data['relation']}\". Generate the region of the object. {concise_post}""",
            f"""Based on the relationship and the subject, identify the object region. Subject region: {region} Relationship: {meta_data['relation']} {concise_post}"""
        ]
    elif task == 'descriptive_object_region_generate':
        instructs = [f"""Given the description of an object, generate the region that contains this object. The description is: \"{text}\" {concise_post}""",
                    f"""In this task, you are required to identify the object that is described by \"{text}\" and output the region of that object. {concise_post}""",
                    f"""What is the region of the object described by \"{text}\" in image? {concise_post}""",
                    f"""Where is the object described by \"{text}\"? {concise_post}""",
                    f"""Find the region of {text}. {concise_post}""",
        ]
    elif task == 'descriptive_object_region_select':
        instructs = [
                    f"""Given the description of an object, select the region that contains this object.\n\nThe description is: \"{text}\"{options_token} {split_token.join(options)}.  {concise_post}""",
                    f"""In this task, you are required to identify the object that is described by \"{text}\" and select the region of that object from options.{options_token} {split_token.join(options)}.  {concise_post}""",
                    f"""What is the region of the object described by \"{text}\" in the picture?{options_token} {split_token.join(options)}.  {concise_post}""",
                    f"""Select the region of the object described by \"{text}\".{options_token} {split_token.join(options)}.  {concise_post}""",
                    f"""Given the image, select the region of {text}.{options_token} {split_token.join(options)}.  {concise_post}"""
        ]
    elif task == 'object_description_generate':
        instructs = [f"Generate a sentence to describe the object in the given bounding box. The description should help people to distinguish the object from other objects in the image.\n\nBounding box: {region_split_token.join(region)}",
                     f"Describe the object in the given region {region_split_token.join(region)}. The description should be about the location and appearance of the object so it can be distinguished from other object in the image.",
                     f"Given the object in {region_split_token.join(region)}, write a sentence to describe it. So it can be easily identified by people. {concise_post}",
                     f"Write a sentence to describe the object in the given region.\n\nRegion: {region_split_token.join(region)}.  {concise_post}",
                     f"Write a description of the object in region: {region_split_token.join(region)}. The description should help people to locate the object without causing confusion."
        ]
    elif task == 'image_quality':
        instructs = [f"Select the reason from options to explain why the image quality is bad. {options_token} {split_token.join(options)} {concise_post}",
                     f"Explain why the image quality is bad. {options_token} {split_token.join(options)} {concise_post}",
                     f"Tell me what is wrong with the image. {options_token} {split_token.join(options)} {concise_post}",
                     f"The image quality might be low. Tell me why. {options_token} {split_token.join(options)} {concise_post}",
                     f"Select a reason for the bad quality of the image. {options_token} {split_token.join(options)} {concise_post}"
                     ]
    elif task == 'text_localization':
        instructs = [
            f"""Select the region from options that contains the given letters: \"{text}\". {options_token} {split_token.join(options)} {concise_post}""",
            f"""Determine which region contains the letters: \"{text}\"? {options_token} {split_token.join(options)} {concise_post}""",
            f"""Select the region that contains the text \"{text}\" {options_token} {split_token.join(options)} {concise_post}""",
            f"""Which region contains \"{text}\" {options_token} {split_token.join(options)} {concise_post}""",
            f"""Identify the region that has \"{text}\" written on. {options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'text_legibility':
        instructs = [
            f"""Look at the given text region of the {image_token} and decide whether the text in the region is clear and complete. {region_token} {split_token.join(region)} {options_token} {split_token.join(options)} {concise_post}""",
            f"""Decide if the text in {split_token.join(region)} is clear and complete. {options_token} {split_token.join(options)} {concise_post}""",
            f"""Decide if the text in the given region is legible. Region {split_token.join(region)} {options_token} {split_token.join(options)} {concise_post}""",
            f"""In this task, you are given a region which has some text written on it. Tell me if the text on that region is clear. Region {split_token.join(region)} {options_token} {split_token.join(options)} {concise_post}""",
            f"""Tell me if the text on {split_token.join(region)} is clear and readable. {options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'text_type':
        instructs = [
            f"""Look at the text in the given region of the {image_token} and determine the type of text in the region from options. {region_token} {split_token.join(region)} {options_token} {split_token.join(options)} {concise_post}""",
            f"""Read the text in {split_token.join(region)} of the {image_token} and select the type of text from options. {options_token} {split_token.join(options)} {concise_post}""",
            f"""What type is the text in {split_token.join(region)}? {options_token} {split_token.join(options)} {concise_post}""",
            f"""The type of the text in {split_token.join(region)} is {options_token} {split_token.join(options)} {concise_post}""",
            f"""look at the text in {split_token.join(region)} and tell me it's type. {options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'region_text_match':
        instructs = [
            f"""Look at the letters in {region_token} {split_token.join(region)} and determine if the letters in the region are the same as \"{text}\". {options_token} {split_token.join(options)} {concise_post}""",
            f"""Is the text \"{text}\" in {split_token.join(region)}? {options_token} {split_token.join(options)} {concise_post}""",
            f"""Does {split_token.join(region)} have the letters \"{text}\"? {options_token} {split_token.join(options)} {concise_post}""",
            f"""Is the text in {split_token.join(region)} the same as \"{text}\"? {options_token} {split_token.join(options)} {concise_post}""",
            f"""Do the letters in {split_token.join(region)} match \"{text}\"? {options_token} {split_token.join(options)} {concise_post}"""
        ]
    elif task == 'multimodal_factual_checking':
        instructs = [
            f"Deicide if the claim can be supported by the image and the context.\n\nContext: {context}\n\nClaim: \"{text}\"{options_token} {split_token.join(options)} {concise_post}",
            f"Context: {context}\nCan the context support \"{text}\"? {options_token} {split_token.join(options)} {concise_post}",
            f"{context}\n\nRead previous text and decide if \"{text}\" is factually correct? {options_token} {split_token.join(options)} {concise_post}",
            f"Does the context support \"{text}\"?\n\nContext: {context} {options_token} {split_token.join(options)} {concise_post}",
            f"Context: {context}\n\nDoes the context support \"{text}\"? {options_token} {split_token.join(options)} {concise_post}"
        ]
    elif task == 'wikihow_next_step':
        context = '\n'.join(context) if len(context)>0 else '\"nothing\"'
        instructs = [
            f"For the task {meta_data['method']}, given the history steps {context} and the current step with its corresponding image, what is the next step for this task? The current step is {text}, what is the next step?",
            f"What is the next step? You are doing {meta_data['method']} and you have finished\n\n{context}\nYou currently are at the step given by the image and the text \"{text}\". The next step is",
            f"You are doing {meta_data['method']}. You have done\n\n{context}\nNow you are at the step described by the image. What is the next step?",
            f"The goal is to \"{meta_data['method']}\". Given current step specified by the content of the image and you have finished.\n\n\All previous steps: {context}.\nWhat is the next step?",
            f"You are doing {meta_data['method']} and you are at \"{text}\" step. The previous steps you finished are\n\n{context}\nWhat is the next step?",
        ]
    elif task == 'wikihow_text_image_step_order':
        options = ['next','previous']
        random.shuffle(options)
        instructs = [
            f"For the task \"{meta_data['method']}\", given the current step, decide if the content of the image is the next or previous step.\nThe current step is {text}.{options_token} {split_token.join(options)}",
            f"Is the image the next or previous step? You are doing \"{meta_data['method']}\" and you are currently at \"{text}\".{options_token} {split_token.join(options)}",
            f"The overall goal is to {meta_data['method']}. You are at \"{text}\" step. Is the image the next or the previous step?{options_token} {split_token.join(options)}",
            f"The goal is to \"{meta_data['method']}\". Given the current step \"{text}\", Is the picture the next or the previous step?{options_token} {split_token.join(options)}",
            f"You are doing {meta_data['method']}. Is the step specified in the picture the next or previous step to \"{text}\"?{options_token} {split_token.join(options)}",
        ]
    elif task == 'wikihow_image_text_step_order':
        options = ['next','previous']
        random.shuffle(options)
        instructs = [
            f"For the task \"{meta_data['method']}\", decide if \"{text}\" is the next or previous step to the step specified by the image.{options_token} {split_token.join(options)}",
            f"Is \"{text}\" the next or previous step? You are doing \"{meta_data['method']}\" and you are currently at the step described by the image.{options_token} {split_token.join(options)}",
            f"The overall goal is to {meta_data['method']}. You are at the step specified by the content of the image. Is \"{text}\" the next or the previous step?{options_token} {split_token.join(options)}",
            f"The goal is to \"{meta_data['method']}\". Given the current step in the picture, Is \"{text}\" the next or the previous step?{options_token} {split_token.join(options)}",
            f"You are doing {meta_data['method']}. Is the step \"{text}\" the next or previous step to the step in the image?{options_token} {split_token.join(options)}",
        ]
    elif task == 'wikihow_immediate_next_step_selection':
        instructs = [
            f"For the task \"{meta_data['method']}\", select the immediate next step to the step specified by the image.{options_token} {split_token.join(options)}",
            f"You are doing \"{meta_data['method']}\" and you are currently at the step described by the image. What is your next step?{options_token} {split_token.join(options)}",
            f"The overall goal is to {meta_data['method']}. You are at the step specified by the content of the image. Select the immediate next step from the options.{options_token} {split_token.join(options)}",
            f"The goal is to \"{meta_data['method']}\". Given the current step in the picture, what is the next step?{options_token} {split_token.join(options)}",
            f"You are doing {meta_data['method']}. What is the next step to step in the image?{options_token} {split_token.join(options)}",
        ]
    elif task == 'image_text_selection':
        instructs = [f"""Select the text from options that best describes the image. {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Which text in the options best describes the image? {options_token} {split_token.join(options)} {concise_post}""",
                     f"""In this task, you are given some sentences and you need to decide which sentence best matches the image.{options_token} {split_token.join(options)} {concise_post}""",
                     f"""Which option in the options that is the caption of the image. {options_token} {split_token.join(options)} {concise_post}""",
                     f"""Select the caption of the image. {options_token} {split_token.join(options)} {concise_post}""",
                     ]
    elif task == 'visual_attribute':
        instructs = [
            f"""Decide which option is the attribute of the object in the given region.\nRegion: {region_split_token.join(region)}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Select the attribute of the object in {region_split_token.join(region)}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Given object in {region_split_token.join(region)}, select its attribute.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Given the region of the object, select its attribute from the options.\n\nRegion: {region_split_token.join(region)}{options_token} {split_token.join(options)}.  {concise_post}""",
            f"""Given the bounding box {region_split_token.join(region)} of the object, select its attribute.{options_token} {split_token.join(options)}.  {concise_post}""",
        ]    
    # image generation tasks
    elif task == 'infilling':
        instructs = [
            f"Fill in the missing part of the image.",
            f"Generate the missing part of the image.",
            f"Generate masked part of the image.",
            f"Generate the part of the image covered by the black square.",
            f"Generate the part of the image covered by black.",
        ]
    elif task == 'im_region_extraction':
        instructs = [
            f"Extract part of the image specified by the given region. Region: {region}.",
            f"Extract the part of image in {region}",
            f"Generate a copy of the image in the given region {region}.",
            f"Output a new image that is identical to the part of the given image specified by {region}",
            f"Generate a new image that is a precise replica of the area {region} in the given image.",
        ]
    elif task == 'im_descriptive_infilling':
        instructs = [
            f"Fill in the missing part of the image based on the description \"{text}\".",
            f"Generate the missing part of the image. The caption of the missing part is \"{text}\".",
            f"Using the caption \"{text}\" to generate the region of the image covered by black.",
            f"Based on the description \"{text}\", generate the masked part in the current image.",
            f"Generate the image that fills in the black square in the given image. The description of the black square is \"{text}\".",
        ]
    elif task == 'image_completion_w_region_caption':
        instructs = [
            f"Fill in the missing part of the image based on the description \"{text}\" and output the whole image.",
            f"Base on the caption \"{text}\", fill in the missing part of the image and generate the complete image.",
            f"Generate a full version of the given image using the caption to fill in the black area. Caption: {text}.",
            f"Create a new image based on the original, with the missing area filled in by \"{text}\".",
            f"Generate a complete version of the image with the missing area filled in. The caption of the missing area is \"{text}\"",
        ]
    elif task == 'image_completion_w_image_caption':
        instructs = [
            f"Complete the image based on the description \"{text}\".",
            f"Generate an image with description \"{text}\" by filling in the black area in the given image",
            f"Use the provided caption to produce a complete image by filling in the black area. Caption: \"{text}\"",
            f"Generate a new image that is the same as the given image with the missing area filled. Caption for the new image is \"{text}\".",
            f"Use the given caption to generate a new image based on the given image with the masked part filled in. Caption: \"{text}\".",
        ]
    
    elif task == 'VQA_activity_recognition':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you will answer a question about the activity of an object in the image. The question is "{question}"{options_token} {split_token.join(options)}. {concise_post}""",
            f"""You are asked about the activity of animals or people in the image. Look at the image and answer "{question}" You should select your answer from the given options.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question} Answer the question by first finding the object in the image and identify its activity. The answer is in the options.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you will be asked about the activity of some object in the image. Select the best answer from options. Question: {question}{options_token} {split_token.join(options)}. {concise_post}"""
        ]
    elif task == 'VQA_attribute':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you will be asked a question about the attribute of an object in the image. The question is "{question}"{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Answer the following question about the attribute of an object, "{question}" Select your answer from the given options.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question}\n\nAnswer above question by first finding the object in the image and select its attribute from options.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you will be asked about the attribute of some object. Select the best answer from given options. Question: {question}{options_token} {split_token.join(options)}. {concise_post}"""
        ]
    elif task == 'VQA_color':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you are asked the color of some object in the image. Question: {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question}\n\nAnswer the above question by first finding the object in the image and then select its color from options,{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Answer {question} based on the image. {options_token} {split_token.join(options)}. {concise_post}""",
            f"""Answer the question: "{question}" based on the color of an object. {concise_post}"""
        ]
    
    elif task == 'VQA_counting':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you are asked a question about the number of some objects in the image. The question is: {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""The question is: {question} Select your answer from options.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question}\n\nPlease answer the question by counting the object mentioned in the question.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""This task tests your ability to count number of objects. Here is the question "{question}". Select the correct answer from options.{options_token} {split_token.join(options)}. {concise_post}"""
        ]
    
    elif task == 'VQA_object_presence':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""This task asks you to identify if an object appears in the image. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you are required to answer a question about the appearance of an object.{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""{question} Decide if the object mentioned in previous question appears in the image.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question} look at the image and answer the question.{options_token} {split_token.join(options)}. {concise_post}""",
        ]
        
    elif task == 'VQA_object_recognition':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task you are asked a question about the type of an object in the image. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you will answer a question about the subclass of an object in the image. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you will be presented with an image. Your task is to answer a question about the type of object. Question: {question}{options_token} {split_token.join(options)}.  {concise_post}
            """,
            f"""Please answer a question regarding the type of an object in the image. Question: {question}{options_token} {split_token.join(options)}. {concise_post}"""
            
        ]
    elif task == 'VQA_positional_reasoning':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you need to analyze the position of objects in an image and answer the following question. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""This task requires an understanding of object location within the presented image. Please select the correct answer to the provided question. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, the goal is to understand the location of objects within the presented image and provide a answer to the question provided. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question}{options_token}\n\n Please answer the question by reasoning about the positions of objects and select an answer from options. {split_token.join(options)}. {concise_post}"""
        ]
        
    elif task == 'VQA_scene_recognition':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you need to pay attention to the scene in the image and answer the following question.\n {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question}{options_token}. \n Please answer the question by analyzing the scene in the provided image. Here are some possible answers. {options_token} {split_token.join(options)}. {concise_post}""",
            f"""Look at the environment in the image and answering the question accordingly.\n {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Given a picture of certain environment, answer the following question by select an answer from the options. \n {question}{options_token} {split_token.join(options)}. {concise_post}"""
        ]
        
    elif task == 'VQA_sentiment_understanding':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""This task requires an understanding of the feeling conveyed in the image. Please select the correct answer to the provided question. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question}{options_token} {split_token.join(options)}.\n Please answer the question by interpreting the sentiment in the image. {concise_post}""",
            f"""Please analyze the sentiment depicted in the image and answer the question.\n {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you will be asked a question regarding the emotion conveyed in the image. The question is {question}{options_token} {split_token.join(options)}. {concise_post}"""
        ]
        
    elif task == 'VQA_sport_recognition':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you need to pay attention to the sports depicted in the image and answer the following question. \n {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Given a picture about sports, answer the following question by select an answer from the options. \n {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""There are some sports taking place in the image. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Please answer the following question by analyzing the sport in the given image.\n {question}{options_token} {split_token.join(options)}. {concise_post}"""
        ]
        
    elif task == 'VQA_utility_affordance':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you need to pay attention to the possible actions can be taken to the objects in the image and answer the following question. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Please take a look at the picture and answer the following question by thinking about what each object in the picture can be used for. {question}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Question: {question}{options_token}\n Please select a correct answer for the question by analyzing the affordance of the objects in the image. {split_token.join(options)}. {concise_post}""",
            f"""This task tests your ability to understand the potential actions that you can take on the objects or the usage of the objects in the image. Here is the question "{question}". Select the correct answer from options.{options_token} {split_token.join(options)}. {concise_post}"""
        ]
        
    elif task == 'select_overlap_most_region':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, decide which region in the options overlaps most with given region.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Select the region that shares the most common area with {given_region}.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Which option overlaps most with {given_region}?{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Decide the region that has the most common area with the given region. Region: {given_region}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Region: {given_region}\n\nIdentify the region overlaps most with the above given region from options.{options_token} {split_token.join(options)}. {concise_post}"""
        ]
    elif task == 'select_overlap_least_region':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, decide which region in the options shares the least common area with given region.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""In this task, you are given a region: {given_region}, you need to select a region from the options that has the least overlap with the given region.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Which option has the least shared area with {given_region}?{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Select the region that has the least overlap with {given_region}.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Given region: {given_region}, decide which option has the least common area with it.{options_token} {split_token.join(options)}. {concise_post}""",
        ]
    elif task == 'select_overlaped_region':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, select an overlapping region from options.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Select a region from options that overlaps with {given_region}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Which region from options that shares common area with {given_region}?{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Region: {given_region}\n\nSelect a region that has overlap with the given region.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Which region from options that has common area with {given_region}?{options_token} {split_token.join(options)}. {concise_post}""",
            
        ]
    elif task == 'select_nonoverlaped_region':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, select an non-overlapping region from options.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Region: {given_region}, select an non-overlapping region with the given region from options.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Select an option that does not overlap with {given_region}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Which option does not share common area with {given_region}?{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Tell me which option does not have shared area with {given_region}?{options_token} {split_token.join(options)}. {concise_post}"""
        ]
    elif task == 'if_region_overlap':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, decide if {region[0]} overlaps with it.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Do the following two regions overlap? Region 1: {region[0]} and Region 2: {given_region}{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Does {given_region} share common area with {region[0]}?{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Tell me if {region[0]} and {given_region} have common area.{options_token} {split_token.join(options)}. {concise_post}""",
            f"""Do {region[0]} and {given_region} overlap?{options_token} {split_token.join(options)}. {concise_post}"""
        ]
    
    # ----------------------- testing tasks ----------------------------- #
    
    elif task == 'visual_nli':
        instructs = [
            f"""In this task, you are given a short sentence and a image. You need to decide if the content of the image can support the text. Text: {text} {options_str}. {concise_post}""",
            f"""Can you conclude \"{text}\" from the content of image? Select your answer from the options.{options_str}. {concise_post}""",
            f"""Can you conclude \"{text}\" from the content of image?{options_str}. {concise_post}""",
            f"""Can the content of the image support \"{text}\"?{options_str}. {concise_post}""",
            f"""Does the image support the given text?\nText: {text}{options_str}. {concise_post}"""
        ]
    elif task == 'natural_language_visual_reasoning':
        instructs = [
            f"""Does the content of the image support the given text? Text: {text}{options_str}. {concise_post}""",
            f"""Based on the image, is \"{text}\" true?{options_str}. {concise_post}""",
            f"""Decide if the content of the image supports the sentence. Sentence:{text}{options_str}. {concise_post}""",
            f"""\"{text}\"\n\nIs the above text true based on the picture?{options_str}. {concise_post}""",
            f"""Look at the picture and is \"{text}\" true?{options_str}. {concise_post}""",
        ]
    elif task == 'visual_spatial_reasoning':
        instructs = [
            f"""The text is about the spatial relationship between two objects in the image. If the text is true?\n\nText{text}{options_str}. {concise_post}""",
            f"""In this task, you need to decide if the positional relationship in the text is true, based on the image. The text is: {text}{options_str}. {concise_post}""",
            f"""Does the content of the image support the sentence below?\n\nSentence:{text}{options_str}. {concise_post}""",
            f"""Can the image support \"{text}\"?{options_str}. {concise_post}""",
            f"""Is \"{text}\" true, by referring to the image?{options_str}. {concise_post}""",
        ]
    elif task == 'commonsense_VQA':
        region_info= ' '.join([ f"{k} is in {v[0]}."  for k, v in meta_data['object_regions'].items()])
        if instruction_id >=2:
            for k, v in meta_data['object_regions'].items():
                if k in question:
                    question = question.replace(k, f"the {k} in {' '.join(v)}")
        instructs = [
            f"""{question} {region_info}{options_str}""",  
            f"""The region information: {region_info}\nBased on the region information and the image, {question}{options_str}""",
            f"""{question}{options_str}""",
            f"""Look at the image and the regions in the question, {question}{options_str}""",
            f"""Based on the image, answer {question}{options_str}""",
        ]
    # ----------------------------------------- VQA
    elif task == 'text_vqa':
        instructs = [
            f"""Based on the image and the text on the image, answer the question below.\n\n{question}. {concise_post}""",
            f"""There is some text on the image. Answer {question} based on the text in image. {concise_post}""",
            f"""Look at the text on image and answer: {question}. {concise_post}""",
            f"""Look at the image and {question}. {concise_post}""",
            f"""{question}. {concise_post}""",
        ]
    elif task == 'grounded_VQA':
        instructs = [
            f"""In this task, you are given a question and you need to identify a region from options in order to answer it. The question is: {question}{options_str}. {concise_post}""",
            f"""Select a region from the options to answer \"{question}\"{options_str}. {concise_post}""",
            f"""Which region can be used to answer \"{question}\"{options_str}. {concise_post}""",
            f"""Which region is the answer to \"{question}\"{options_str}. {concise_post}""",
            f"""{question}{options_token} {split_token.join(options)}. {concise_post}""",
        ]
    
    elif task == 'ok_vqa':
        # instructs = [
        #     f"""In this task, you will be asked a question about image. However, in order to answer this question, you need knoweldge outside of this image. The question is: {question}""",
        #     f"""Question: {question}\n\nUse external knoweldge and the content of the image to answer the question.""",
        #     f"""Based on the content of the image and external knowledge, {question}""",
        #     f"""Based on your knowledge, {question}""",
        #     f"""{question}""",
        # ]
        raise NotImplementedError
        # instructs = [
        #     f"""Answer \"{question}\" based on the content of image and external knowledge.""",
        #     f"""Based the image and background knowledge, {question}""",
        #     f"""Use external knoweldge and the content of the image to answer: {question}""",
        # ]
    elif task == 'ocr':
        instructs = [
            f"""What is the text in the given region {region[0]} in the {image_token}? {concise_post}""",
            f"""Look at image and answer what is the text in {region[0]}? {concise_post}""",
            f"""In this task, you are require to extract the text in a region of the image. The region is {region[0]}. What is the text? {concise_post}""",
            f"""What is the text in the given region of the {image_token}. {region_token} {split_token.join(region)}. {concise_post}""",
            f"""What is the text written on {split_token.join(region)} of the image. {concise_post}""",
        ]
        
        
        
    elif task == 'visual_answer_justification':
        # TODO:
        region_info= ' '.join([ f"{k} is in {v[0]}."  for k, v in meta_data['object_regions'].items()])
        
        instructs = [
            f"""Given the image and question: \"{question}\"\nThe regions of the objects are: {region_info} Select an explanation from options to exlpain why \"{answer}\" is the answer.{options_str}. {concise_post}""",
            f"""{region_info}\n\nGiven the question \"{question}\" Why \"{answer}\" is the answer?{options_str}. {concise_post}""",
            f"""{region_info}\n\nWhy \"{answer}\" is the answer to the question \"{question}\"? {options_str}. {concise_post}""",
            f"""Why \"{answer}\" is the answer to \"{question}\"?\nThe regions are {region_info}{options_str}. {concise_post}""",
            f"""Given the image and question: \"{question}\" {region_info} Select an explanation from options to exlpain why \"{answer}\" is the answer.{options_str}. {concise_post}""",
        ]
    # ------------------------------- misc
    elif task == 'visual_dialog':
        dial_history = [ f"{dial_turn['q']}, {dial_turn['a']};" for dial_turn in meta_data['dialog_hist']]
        if len(dial_history) > 0:
            instructs = [
                f"""Dialogue history: {' '.join(dial_history)}\nBased on the image and dialogue history, answer: {question}? {concise_post}""",
                f"""Context: {' '.join(dial_history)}\n\nGiven the image, answer the question.\n\nQuestion: {question}? {concise_post}""",
                f"""Given the image and the dialog history below:\n\n{' '.join(dial_history)}\n{question}? {concise_post}""",
                f"""Context: {dial_history[-1]}\n\nBased on the image, answer {question}? {concise_post}""",
                f"""{question}? {concise_post}"""
            ]
        else:
            instructs = [
                f"""Given the image, answer the question.\n\nQuestion: {question}? {concise_post}""",
                f"""Based on the image, answer: {question}? {concise_post}""",
                f"""Based on the image, answer {question}? {concise_post}""",
                f"""Given the image, {question}? {concise_post}""",
                f"""{question}? {concise_post}"""
            ]
    elif task == 'purpose_driven_affordance': # remove
        # instructs = [
        #     f"""Given the image what can you do to the object in {region[0]}?""",
        #     f"""What does {region[1]} do to the object in {region[0]}?"""
        # ]
        raise NotImplementedError
    elif task == 'visual_text_extraction':
        instructs = [
                     f"""This image contains some text. For this task, you need to look at the image carefully and identify all the text in the image. The text in the image is""",
                     f"""There is some text written on the image. Tell me what is the text.""",
                     f"""What is the text written on the image?""",
                     f"""The text written on the image is""",
                     f"""Tell me all the text on the image."""
        ]
    elif task == 'hateful_content_detection':
        instructs = [
                     f"""In this task, you need to decide if there is hateful content in the given image. The image itself may not contain hateful content but when combined with the text written on the image, it may have.{options_str}""",
                     f"""Considering both the content of the image and the text on the image, decide if it contains hateful intension.{options_str}. {concise_post}""",
                     f"""Look at the image and the text on the image. Decide if there is hateful intention.{options_str}. {concise_post}""",
                     f"""Decide if there is hateful content in the given image.{options_str}. {concise_post}""",
                     f"""Is there hateful intention in the given image?{options_str}. {concise_post}""",
        ]
    elif task == 'medic_damage_severity':
        instructs = [
            f"""What is the damage level in the image? {concise_post}""",
            f"""Look at the image and decide the damage level. The damage level is given in options.{options_str}. {concise_post}""",
            f"""Select the damage severity from options.{options_str}. {concise_post}""",
            f"""In this task, you are required to decide how bad is the damage in the image. The levels of damage are severe, mild, and little or none.  {concise_post}""",
            f"""Tell me how bad is the damage in the image.{options_str}. {concise_post}"""
        ]
    elif task == 'medic_informative':
        instructs = [
            f"""Does this image provide any information about the disaster? Choose the correct answer from options.{options_str}. {concise_post}""",
            f"""Is this picture informative about a disaster?{options_str}. {concise_post}""",
            f"""Is this a informative picture of disaster?{options_str}. {concise_post}""",
            f"""Can you gain any information about a disaster from the image? If yes, select informative, if not select not informative from the options.{options_str}. {concise_post}""",
            f"""If this picture is about a disaster, select informative. Otherwise, select not informative from options.{options_str}. {concise_post}""",
        ]
    elif task == 'medic_disaster_types':
        instructs = [
            f"""According to the image, what kind of disaster happened? Choose the correct answer from options.{options_str}. {concise_post}""",
            f"""What kind of disaster happens in the image? If no disaster happens in the image, select not disaster.{options_str}. {concise_post}""",
            f"""What disaster happens in the image?{options_str}. {concise_post}""",
            f"""Based on the image, what is the disaster. Select your answer from options.{options_str}. {concise_post}""",
            f"""Look at the image and tell me what kind of disater is in the image. If no disaster, select not disaster.{options_str}. {concise_post}""",
        ]
    elif task == 'image_generation': 
        instructs = [
            f"""what is the complete image? caption: {text}.""", # ofa instruction
            f"""Generate an image with the caption \"{text}\".""",
            f"""Create an image of \"{text}\"."""
            f"""Generate the image that corresponds to the description \"{text}\"."""
            f"""Generate the missing part of the image, based on the text \"{text}\"."""
        ]
    elif task == 'im_descriptive_extraction':
        instructs = [
            f"""Extract the part of the image with caption \"{text}\"""", 
            f"""Extract part of the image specified by the given caption. Caption: \"{text}\"""",
            f"""Given an image, generate a copy of the image with \"{text}\" as description.""",
            f"""Find the region that most accurately depicts \"{text}\" and then create an image of that region.""",
            f"""Create a new image by extracting the region that most accurately corresponds to \"{text}\" in the original image."""
        ]
    else:
        raise NotImplementedError
    
    assert len(instructs) >= 5, f"task {task} has only {len(instructs)} instructions"
    
    return random.choice(instructs), target