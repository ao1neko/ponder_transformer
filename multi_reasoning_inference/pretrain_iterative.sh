#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=iterative \
        --train_dataset_name="./numerical_datasets/data/pretrain/pretrain_iterative.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/pretrain/pretrain_iterative.jsonl"  \
        --test_dataset_name="./numerical_datasets/data/pretrain/pretrain_test_iterative.jsonl"  \
        --model_name=t5 \
        --load_model_dir="t5-base" \
        --train=true \
        --predict=false \
        --train_epochs=100 \
        --eval_steps=100  \
        --save_steps=100 \
        --output_dir="/work02/aoki0903/inference_model/save/pretrain/iterative" \
        --run_dir="save/pretrain/iterative" \
        --batch_size=64 \
        --learning_rate=0.001