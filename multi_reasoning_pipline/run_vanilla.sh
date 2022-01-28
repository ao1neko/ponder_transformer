#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python run.py \
    --json_pass=$WOKE01/datas/ \
    --epochs=200 \
    --max_step=6 \
    --batch_size=128 \
    --seed=6 \
    --print_sample_num=3 \
    --valid=true \
    --load_pass=best_vanilla_models_shuffle/deco6_num \
    --log_dir=runs_shuffle/vanilla/deco6_num \
    --ponder_model=false \
    --train=true \
    --device=cuda:1 \
    --emb_dim=512 \
    --lr=1.0

#./run_vanilla.sh depth3
