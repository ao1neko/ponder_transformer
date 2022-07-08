#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

NUMERICAL_DIR="/home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_pipline/data_generater"
WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"
max_value=$1

PYTHONHASHSEED=0 python $NUMERICAL_DIR/make_drop_format_data.py \
    $WOKE01/datas/pretrain_data_$max_value.json \
    --pretrain=true \
    --pretrain_max_value=$max_value \
    
echo "make drop format pretrain data"

#./run_make_pretraindata.sh 500