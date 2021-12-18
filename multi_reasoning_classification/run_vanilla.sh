#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python run.py \
    --json_pass=$WOKE01/datas/$1.json \
    --epochs=100 \
    --max_step=10 \
    --batch_size=128 \
    --seed=6 \
    --print_sample_num=3 \
    --valid=true \
    --load_pass=best_vanilla_models_shuffle/mod_depth3 \
    --log_dir=runs_shuffle/vanilla/mod_depth3 \
    --ponder_model=false \
    --loop_model=false \
    --device=cuda:0 \
    --concated=false \
    --emb_dim=128 \
    --lr=0.0001

#./run_vanilla.sh depth3
