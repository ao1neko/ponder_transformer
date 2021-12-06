#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python run.py \
    --json_pass=$WOKE01/datas/$1.json \
    --epochs=3000 \
    --max_step=1 \
    --batch_size=128 \
    --seed=6 \
    --print_sample_num=3 \
    --valid=true \
    --load_pass=best_vanilla_models_mul3_sizeconst_small/depth3 \
    --log_dir=runs_mul3_sizeconst_small/vanilla/depth3 \
    --ponder_model=false \
    --device=cuda:2 \
    --emb_dim=1024 


#./run_vanilla.sh ponder_base
#./run_vanilla.sh multi_reasoning_depth3_mul

#./run_vanilla.sh multi_reasoning_depth4_mul3_sizeconst_small
#./run_vanilla.sh multi_reasoning_depth6_mul3
