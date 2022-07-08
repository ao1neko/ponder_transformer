#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --datasets_name=proofwriter_iterative \
        --train_dataset_name="./proofwriter-dataset-V2020.12.3/convert-CWA/depth-0-2/meta-train.jsonl" \
        --valid_dataset_name="./proofwriter-dataset-V2020.12.3/convert-CWA/depth-0-2/meta-dev.jsonl" \
        --test_dataset_name="./proofwriter-dataset-V2020.12.3/convert-CWA/depth-0-2/meta-test.jsonl" \
        --model_name=t5 \
        --load_model_dir="t5-base" \
        --train=true \
        --predict=true \
        --train_epochs=1000 \
        --eval_steps=2000  \
        --output_dir="/work02/aoki0903/dentaku_model/save/proofwriter_iterative_0_2" \
        --run_dir="save/proofwriter_iterative_0_2" \
        --batch_size=4 \
        --learning_rate=0.0001


#        --output_dir="save/test_proofwriter_at_once" \
