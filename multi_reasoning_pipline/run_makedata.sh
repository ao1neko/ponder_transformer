#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

NUMERICAL_DIR="/home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_pipline/data_generater"
WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"

PYTHONHASHSEED=0 python $NUMERICAL_DIR/number_file_generator.py \
    $NUMERICAL_DIR/number_configs/number_config.jsonnet

declare -a method=("train" "valid" "test")    # 初期化
declare -a question_num=($2 $3 $4)    # 初期化

for i in {0..2}; do
PYTHONHASHSEED=0 python  $NUMERICAL_DIR/copy_jsonfile.py \
    $NUMERICAL_DIR/configs/$1.json \
    $NUMERICAL_DIR/configs/$1_${method[$i]}.json \
    $WOKE01/datas/"${method[$i]}_numbers.pkl"

echo "copied ${method[$i]}_config_file"

PYTHONHASHSEED=0 python $NUMERICAL_DIR/make_drop_format_data.py \
    $WOKE01/datas/$1_${method[$i]}"_"${question_num[$i]}.json \
    --config_filepath=$NUMERICAL_DIR/configs/$1_${method[$i]}.json \
    --max_number_of_question=${question_num[$i]}
echo "make drop format ${method[$i]} data"
done
#./run_makedata.sh arg2 1000