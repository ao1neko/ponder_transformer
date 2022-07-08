#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=ansonly \
        --train_dataset_name="./numerical_datasets/data/depth_1_num_1/train_ansonly.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_1_num_1/valid_ansonly.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_1_num_1/test_ansonly.jsonl" \
        --model_name=t5 \
        --load_model_dir="t5-base" \
        --train=true \
        --predict=false \
        --train_epochs=1 \
        --eval_steps=1  \
        --output_dir="/work02/aoki0903/inference_model/save/depth_1_num_1/ansonly" \
        --run_dir="save/depth_1_num_1/ansonly" \
        --batch_size=4 \
        --learning_rate=0.0001


#        --output_dir="save/test_proofwriter_at_once" \
