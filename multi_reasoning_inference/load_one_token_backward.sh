#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=one_token \
        --train_dataset_name="./numerical_datasets/data/depth_5_distractor_3/train_one_token_backward.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_5_distractor_3/valid_one_token_backward.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_4_distractor_3/test_one_token_backward.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/depth_1_5_distractor_3/one_token_backward/checkpoint-110000" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=100  \
        --output_dir="/work02/aoki0903/inference_model/save/depth_4_distractor_3_by_depth_1_5_distractor_3/one_token_backward" \
        --run_dir="save/depth_4_distractor_3_by_depth_1_5_distractor_3/one_token_backward" \
        --batch_size=8

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
