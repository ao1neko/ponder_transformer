#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=iterative \
        --train_dataset_name="./numerical_datasets/data/depth_1_10_distractor_3/train_iterative_forward_backtrack.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_1_10_distractor_3/valid_iterative_forward_backtrack.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_1_10_distractor_3/test_iterative_forward_backtrack.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/pretrain/iterative/checkpoint-900" \
        --train=true \
        --predict=false \
        --train_epochs=30 \
        --eval_steps=2000  \
        --save_steps=10000 \
        --output_dir="/work02/aoki0903/inference_model/save/depth_1_10_distractor_3/iterative_forward_backtrack" \
        --run_dir="save/depth_1_10_distractor_3/iterative_forward_backtrack" \
        --batch_size=8 \
        --learning_rate=0.0001


#        --output_dir="save/test_proofwriter_at_once" \