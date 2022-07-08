#!/bin/bash
export PYENV_VERSION=PonderNet

inference_step=5
equation_num=10

PYTHONHASHSEED=0 python make_pretrain_data.py \
        --seed=10 

echo "create pretrain data"