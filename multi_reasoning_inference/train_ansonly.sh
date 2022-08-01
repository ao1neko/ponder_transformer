#!/bin/bash
export PYENV_VERSION=PonderNet

#./train.sh pretrain
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python main.py \
        --device=cuda \
        --architecture_name=ansonly \
        --train_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/train_ansonly.jsonl" \
        --valid_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/valid_ansonly.jsonl" \
        --test_dataset_name="./numerical_datasets/data/depth_1_5_distractor_3/test_ansonly.jsonl" \
        --model_name=t5 \
        --load_model_dir="/work02/aoki0903/inference_model/save/pretrain/ansonly/checkpoint-600" \
        --train=true \
        --predict=false \
        --train_epochs=100 \
        --eval_steps=1000  \
        --save_steps=5000 \
        --output_dir="/work02/aoki0903/inference_model/save/depth_1_5_distractor_3/ansonly" \
        --run_dir="save/depth_1_5_distractor_3/ansonly" \
        --batch_size=16 \
        --learning_rate=0.0001


#        --output_dir="save/test_proofwriter_at_once" \
#        --load_model_dir="/work02/aoki0903/inference_model/save/pretrain/ansonly/checkpoint-600" \
# --load_model_dir="/work02/aoki0903/inference_model/save/pretrain/ansonly_non_t5_pretrain/checkpoint-149000
