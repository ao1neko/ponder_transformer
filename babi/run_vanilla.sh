#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

WOKE="/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/babi/"

PYTHONHASHSEED=0 python run.py \
    --data_pass=$WOKE/$1.txt \
    --epochs=300 \
    --max_step=1 \
    --batch_size=128 \
    --seed=6 \
    --print_sample_num=3 \
    --valid=true \
    --log_dir=runs/vanilla/step5 \
    --ponder_model=false \
    --device=cuda:1 

#./run_vanilla.sh task_1