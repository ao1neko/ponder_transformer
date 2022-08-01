#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=iterative \
        --train_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/train_iterative_forward_backtrack.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/valid_iterative_forward_backtrack.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_14_distractor_3/test_iterative_forward_backtrack.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/depth_1_5_distractor_3/iterative_forward_backtrack/checkpoint-195000" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=100  \
        --output_dir="/work02/aoki0903/inference_model/save/depth_14_distractor_3_by_depth_1_5_distractor_3/iterative_forward_backtrack" \
        --run_dir="save/depth_14_distractor_3_by_depth_1_5_distractor_3/iterative_forward_backtrack" \
        --batch_size=8

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
