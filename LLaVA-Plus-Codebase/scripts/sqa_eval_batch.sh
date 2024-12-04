#!/bin/bash

CHUNKS=8
for IDX in {0..7}; do
<<<<<<< HEAD
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_science \
=======
    CUDA_VISIBLE_DEVICES=$IDX python -m llava_plus.eval.model_vqa_science \
>>>>>>> tool_server_develop
        --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
        --question-file ~/haotian/datasets/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
        --image-folder ~/haotian/datasets/ScienceQA/data/scienceqa/images/test \
        --answers-file ./test_llava-13b-chunk$CHUNKS_$IDX.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode llava_v1 &
done
