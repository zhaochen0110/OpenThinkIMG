import os,sys,torch

from .utils import *

import json, re
from copy import deepcopy


def answer_sequence_to_str(answer_sequence):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"Step {idx+1}. {step['text']}\n\n")
    res_str = "".join(res)
    return res_str

def answer_sequence_to_shepherd_str(answer_sequence,step_tag = 'ки'):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"Step {idx+1}: {step['text']} {step_tag}\n")
    res_str = "".join(res)
    return res_str

def answer_sequence_to_reasoneval_list(answer_sequence):
    res = []
    for idx,step in enumerate(answer_sequence):
        res.append(f"{idx+1}. {step['text']}")
    return res
    


def score_list_to_str(score_list):
    valid2_list = [str(round(i,2)) for i in score_list]
    res =  ", ".join(valid2_list)
    return res


def clean_str(input_str):
    res_str = deepcopy(input_str)
    res_str = re.sub(r'\\+([^\\\s])', r'\\\\\1', res_str)
    res_str = re.sub(r'\\+([\s])', r'\\\\\\\\\1', res_str)
    return res_str

def remove_comments_from_json(json_string):
    """
    移除 JSON 字符串中的单行和多行注释。
    """

    # 匹配 // 和 # 开头的注释，并移除
    return re.sub(r'//.*?$|#.*?$', '', json_string, flags=re.MULTILINE)

def extract_nested_json(text):
    """
    提取嵌套大括号内的 JSON 数据，移除注释后解析。
    Args:
        text (str): 包含 JSON 的文本。
    Returns:
        dict or list or None: 解析成功返回 JSON 数据，失败返回 None。
    """
    stack = []  # 用来记录大括号的匹配
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if not stack:  # 当栈为空时，记录第一个大括号的位置
                start = i
            stack.append("{")  # 压栈
        elif char == "}":
            stack.pop()  # 出栈
            if not stack:  # 当栈为空时，表示找到完整 JSON
                try:
                    # 提取完整 JSON 字符串
                    json_str = text[start:i+1]
                    # 移除注释
                    json_cleaned = remove_comments_from_json(json_str)
                    # 尝试解析为 JSON 对象
                    return json.loads(json_cleaned)
                except json.JSONDecodeError as e:
                    continue  # 如果解析失败，跳过并继续查找
    return None  # 如果未找到完整 JSON，则返回 None

def process_policy_lm_evaluation_response(response):
    """ process the response STRING from the language model"""
    try:
        json_object = extract_nested_json(response)
        assert json_object is not None
        assert "validity" in json_object and "redundancy" in json_object
        return json_object
    except :
        print(f"Invalid JSON Str, response: {response}")
        return None


def remove_step_prefix(text):
    """
    去掉以 'Step x. ' 或 'step x. ' 或 'x. ' 开头的部分，其中 x 是数字
    """
    text = text.strip()
    return re.sub(r"^(Step\s*\d+\.\s*|\d+\.\s*)", "", text, flags=re.IGNORECASE)

def find_subsequence(tensor, subsequence):
    """
    在张量中定位子串的位置。

    Args:
        tensor (torch.Tensor): 主张量。
        subsequence (torch.Tensor): 子串张量。

    Returns:
        List[int]: 子串在主张量中的起始位置索引列表。
    """
    main_len = tensor.size(0)  # 主张量的长度 (假设是二维张量，取列数)
    sub_len = subsequence.size(0)  # 子串的长度

    positions = []  # 存储匹配的起始位置
    for i in range(main_len - sub_len + 1):  # 滑动窗口遍历
        # 比较切片是否与子串相等
        if torch.equal(tensor[i:i+sub_len], subsequence):
            positions.append(i)
    return positions

