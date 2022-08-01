#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=at_once \
        --train_dataset_name="./numerical_datasets/data/pretrain/pretrain_at_once.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/pretrain/pretrain_at_once.jsonl" \
        --test_dataset_name="./numerical_datasets/data/pretrain/pretrain_at_once.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/pretrain/at_once/checkpoint-500" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=1  \
        --output_dir="/work02/aoki0903/inference_model/save/pretrain/at_once_test" \
        --run_dir="save/pretrain/at_once_test" \
        --batch_size=64

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
