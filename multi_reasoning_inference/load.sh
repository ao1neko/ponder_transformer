#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --datasets_name=proofwriter_iterative \
        --train_dataset_name="./proofwriter-dataset-V2020.12.3/CWA/depth-0/meta-train.jsonl" \
        --valid_dataset_name="./proofwriter-dataset-V2020.12.3/CWA/depth-0/meta-dev.jsonl" \
        --test_dataset_name="./proofwriter-dataset-V2020.12.3/CWA/depth-5/meta-test.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/dentaku_model/save/proofwriter_iterative_0_2/checkpoint-138000" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=1  \
        --output_dir="/work02/aoki0903/dentaku_model/save/test_proofwriter_iterative_5_by_0_2" \
        --run_dir="save/test_proofwriter_iterative_5_by_0_2" \
        --batch_size=8

#--load_model_dir="save/proofwriter_iterative/checkpoint-448000" \
