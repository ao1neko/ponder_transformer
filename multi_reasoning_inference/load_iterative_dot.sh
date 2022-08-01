#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=iterative \
        --train_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/train_iterative_dot.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/valid_iterative_dot.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_8_distractor_3/test_iterative_dot.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/depth_1_5_distractor_3/iterative_dot/checkpoint-30000" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=100  \
        --output_dir="/work02/aoki0903/inference_model/save/depth_8_distractor_3_by_depth_1_5_distractor_3/iterative_dot" \
        --run_dir="save/depth_8_distractor_3_by_depth_1_5_distractor_3/iterative_dot" \
        --batch_size=8

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
