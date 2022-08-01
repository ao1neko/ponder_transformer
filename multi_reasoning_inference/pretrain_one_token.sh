#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=one_token \
        --train_dataset_name="./numerical_datasets/data/pretrain/pretrain_one_token.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/pretrain/pretrain_one_token.jsonl"  \
        --test_dataset_name="./numerical_datasets/data/pretrain/pretrain_test_one_token.jsonl"  \
        --model_name=t5 \
        --load_model_dir="t5-base" \
        --train=true \
        --predict=false \
        --train_epochs=100 \
        --eval_steps=1000  \
        --save_steps=1000 \
        --output_dir="/work02/aoki0903/inference_model/save/pretrain/one_token" \
        --run_dir="save/pretrain/one_token" \
        --batch_size=64 \
        --learning_rate=0.001