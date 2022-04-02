#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

NUMERICAL_DIR="/home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_mymodel"
WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"
config_name=$1

PYTHONHASHSEED=0 python $NUMERICAL_DIR/make_data.py \
    $WOKE01/datas/pretrain_data \
    $NUMERICAL_DIR/configs/$config_name \
    --pretrain=true \
    
echo "make drop format pretrain data"

#./run_make_pretraindata.sh pretrain