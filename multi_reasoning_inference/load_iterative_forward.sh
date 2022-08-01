#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=iterative \
        --train_dataset_name="./numerical_datasets/data/depth_5_distractor_1_5/train_iterative_forward.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_5_distractor_1_5/valid_iterative_forward.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_13_distractor_3/test_iterative_forward.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/depth_1_5_distractor_3/iterative_forward/checkpoint-30000" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=100  \
        --output_dir="/work02/aoki0903/inference_model/save/depth_13_distractor_3_by_depth_1_5_distractor_3/iterative_forward" \
        --run_dir="save/depth_13_distractor_3_by_depth_1_5_distractor_3/iterative_forward" \
        --batch_size=8

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
