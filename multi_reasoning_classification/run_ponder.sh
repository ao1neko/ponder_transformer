#!/bin/bash
export PYENV_VERSION=PonderNet

pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python run.py \
    --json_pass=$WOKE01/datas/ \
    --epochs=1000 \
    --max_step=20 \
    --batch_size=128 \
    --beta=0.1 \
    --seed=6 \
    --log_dir=runs_shuffle/ponder/mod_depth13_small \
    --load_pass=best_ponder_models_shuffle/mod_depth13_small.pt \
    --ponder_model=true \
    --loop_model=false \
    --device=cuda:2 \
    --valid=true \
    --train=false \
    --print_sample_num=1 \
    --test=true \
    --concated=true \
    --single=false \
    --emb_dim=128 \
    --lr=0.0001 \
    --lambda_p=18 \



#   --json_pass=$WOKE01/datas/$1.json \
# --json_pass=$WOKE01/datas/
#   --load_pass=best_ponder_models/depth5 \
#  --load_pass=best_ponder_models_mul3_sizeconst_small/depth3.pt \
#    
#./run_ponder.sh depth3