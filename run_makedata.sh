#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

NUMERICAL_DIR="/home/aoki0903/sftp_sync/MyPonderNet/experiment3"
WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

python $NUMERICAL_DIR/make_drop_format_data.py \
    $NUMERICAL_DIR/configs/$1.json \
    $WOKE01/datas/$1.json

#./run_makedata.sh ponder_base