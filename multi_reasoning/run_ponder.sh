#!/bin/bash
export PYENV_VERSION=PonderNet

pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python run.py \
    --json_pass=$WOKE01/datas/$1.json \
    --epochs=300 \
    --max_step=10 \
    --batch_size=128 \
    --beta=0.1 \
    --seed=6 \
    --print_sample_num=3 \
    --valid=true \
    --log_dir=runs/ponder/step5 \
    --ponder_model=true


#./run_ponder.sh ponder_base