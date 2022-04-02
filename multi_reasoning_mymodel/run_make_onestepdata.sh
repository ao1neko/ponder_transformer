#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

NUMERICAL_DIR="/home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_mymodel/data_generater"
DEVELOP_DIR="/home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_mymodel"
WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python $NUMERICAL_DIR/number_file_generator.py \
    $NUMERICAL_DIR/number_configs/number_config.jsonnet

declare -a method=("train" "valid" "test")    # 初期化
   
for i in {0..2}; do
PYTHONHASHSEED=0 python  $NUMERICAL_DIR/copy_jsonfile.py \
    $DEVELOP_DIR/configs/$1.json \
    $DEVELOP_DIR/configs/$1_${method[$i]}.json \
    $WOKE01/datas/"${method[$i]}_numbers.pkl"

echo "copied ${method[$i]}_config_file"

PYTHONHASHSEED=0 python $DEVELOP_DIR/make_data.py \
    $WOKE01/datas/$1_${method[$i]} \
    $DEVELOP_DIR/configs/$1_${method[$i]}.json \
    --onestep=true 
echo "make drop format ${method[$i]} data"
done
#./run_make_onestepdata.sh onestep 