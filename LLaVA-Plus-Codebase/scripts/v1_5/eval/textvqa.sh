#!/bin/bash

<<<<<<< HEAD
python -m llava.eval.model_vqa_loader \
=======
python -m llava_plus.eval.model_vqa_loader \
>>>>>>> tool_server_develop
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

<<<<<<< HEAD
python -m llava.eval.eval_textvqa \
=======
python -m llava_plus.eval.eval_textvqa \
>>>>>>> tool_server_develop
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
