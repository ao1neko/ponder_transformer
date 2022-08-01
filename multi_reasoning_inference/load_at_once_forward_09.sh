#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=at_once \
        --train_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/train_at_once_forward_09.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/valid_at_once_forward_09.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_1_distractor_3/test_at_once_forward_09.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/depth_1_5_distractor_3/at_once_forward_09/checkpoint-10000" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=100  \
        --output_dir="/work02/aoki0903/inference_model/save/depth_1_distractor_3_by_depth_1_5_distractor_3/at_once_forward_09" \
        --run_dir="save/depth_1_distractor_3_by_depth_1_5_distractor_3/at_once_forward_09" \
        --batch_size=8

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
