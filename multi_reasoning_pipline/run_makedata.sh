#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

NUMERICAL_DIR="/home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_pipline/data_generater"
WOKE01="/work01/aoki0903/PonderNet/multihop_experiment"
question_num=$2

PYTHONHASHSEED=0 python $NUMERICAL_DIR/number_file_generator.py \
    $NUMERICAL_DIR/number_configs/number_config.jsonnet

for method in "train" "valid" "test"
do
PYTHONHASHSEED=0 python  $NUMERICAL_DIR/copy_jsonfile.py \
    $NUMERICAL_DIR/configs/$1.json \
    $NUMERICAL_DIR/configs/$1_$method.json \
    $WOKE01/datas/$method"_numbers.pkl"

echo "copied" $method"_config_file"

PYTHONHASHSEED=0 python $NUMERICAL_DIR/make_drop_format_data.py \
    $WOKE01/datas/$1_$method"_"$question_num.json \
    --config_filepath=$NUMERICAL_DIR/configs/$1_$method.json \
    --max_number_of_question=$question_num
echo "make drop format $method data"
done
#./run_makedata.sh arg2 1000