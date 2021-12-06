#!/bin/bash
export PYENV_VERSION=PonderNet

pyenv versions
conda info -e

WOKE="/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/rule-reasoning-dataset-V2020.2.5.0/original"

PYTHONHASHSEED=0 python run.py \
    --data_pass=$WOKE/$1 \
    --epochs=300 \
    --max_step=5 \
    --batch_size=32 \
    --beta=0.01 \
    --seed=6 \
    --print_sample_num=3 \
    --valid=true \
    --log_dir=runs/ponder/depth0 \
    --ponder_model=true \
    --train=true \
    --device=cuda:2


#./run_ponder.sh depth-0