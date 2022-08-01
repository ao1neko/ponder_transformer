#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=one_token \
        --train_dataset_name="./numerical_datasets/data/pretrain/pretrain_one_token.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/pretrain/pretrain_one_token.jsonl" \
        --test_dataset_name="./numerical_datasets/data/pretrain/pretrain_test_one_token.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/pretrain/one_token/checkpoint-19500" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=1  \
        --output_dir="/work02/aoki0903/inference_model/save/pretrain/one_token_test" \
        --run_dir="save/pretrain/one_token_test" \
        --batch_size=64

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
