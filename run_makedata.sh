#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

NUMERICAL_DIR="/home/aoki0903/sftp_sync/MyPonderNet/experiment3"
WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python $NUMERICAL_DIR/number_file_generator.py \
    $NUMERICAL_DIR/configs/number_config.jsonnet

PYTHONHASHSEED=0 python $NUMERICAL_DIR/make_drop_format_data.py \
    $NUMERICAL_DIR/shuffle_configs/$1.json \
    $WOKE01/datas/$1.json 

#./run_makedata.sh ponder_base
#./run_makedata.sh multi_reasoning_depth2
#./run_makedata.sh depth3