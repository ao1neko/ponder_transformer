#!/bin/bash
export PYENV_VERSION=PonderNet

pyenv versions
conda info -e
WOKE="/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/babi/"


PYTHONHASHSEED=0 python run.py \
    --data_pass=$WOKE/$1.txt \
    --epochs=200 \
    --max_step=10 \
    --batch_size=128 \
    --beta=0.01 \
    --seed=6 \
    --print_sample_num=3 \
    --valid=true \
    --log_dir=runs/ponder/step10 \
    --ponder_model=true \
    --device=cuda:1 



#./run_ponder.sh task_1