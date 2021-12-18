#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python run.py \
    --json_pass=$WOKE01/datas/$1.json \
    --epochs=500 \
    --max_step=20 \
    --batch_size=128 \
    --seed=6 \
    --print_sample_num=3 \
    --valid=true \
    --load_pass=best_loop_models_mul3_sizeconst_small/depth6 \
    --log_dir=runs_mul3_sizeconst_small/loop/depth6 \
    --ponder_model=false \
    --loop_model=true \
    --device=cuda:1 \
    --concated=false \
    --emb_dim=128



#./run_loop.sh multi_reasoning_depth6_mul3_sizeconst_small

