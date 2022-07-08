#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --datasets_name=iterative \
        --train_dataset_name="./numerical_datasets/data/depth_1_num_1/train_ansonly.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_1_num_1/train_ansonly.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_1_num_1/train_ansonly.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/depth_1_num_1/ansonly/checkpoint-138000" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=1  \
        --output_dir="/work02/aoki0903//inference_model/save/depth_1_num_1_by_depth_1_num_1/iterative" \
        --run_dir="save/depth_1_num_1_by_depth_1_num_1/iterative" \
        --batch_size=8

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
