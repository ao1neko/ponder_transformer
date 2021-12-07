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
    --beta=0.1 \
    --seed=6 \
    --log_dir=runs_mul3_sizeconst_verysmall/ponder/depth3 \
    --load_pass=best_ponder_models_mul3_sizeconst_verysmall/depth3 \
    --ponder_model=true \
    --device=cuda:0 \
    --valid=true \
    --train=true \
    --print_sample_num=3 \
    --test=true \
    --concated=false \
    --emb_dim=128



#   --json_pass=$WOKE01/datas/$1.json \
# --json_pass=$WOKE01/datas/
#   --load_pass=best_ponder_models/depth5 \
#    
#./run_ponder.sh ponder_base
#./run_ponder.sh multi_reasoning_depth5
#./run_ponder.sh multi_reasoning_depth5_mul
#./run_ponder.sh multi_reasoning_depth3

#./run_ponder.sh multi_reasoning_depth3_mul3_sizeconst_verysmall