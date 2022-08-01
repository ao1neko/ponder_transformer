#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=one_token \
        --train_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/train_one_token_backward.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/valid_one_token_backward.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/test_one_token_backward.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/pretrain/one_token/checkpoint-19500" \
        --train=true \
        --predict=false \
        --train_epochs=50 \
        --eval_steps=2000  \
        --save_steps=10000 \
        --output_dir="/work02/aoki0903/inference_model/save/depth_1_5_distractor_3/one_token_backward" \
        --run_dir="save/depth_1_5_distractor_3/one_token_backward" \
        --batch_size=8 \
        --learning_rate=0.0001


#        --output_dir="save/test_proofwriter_at_once" \