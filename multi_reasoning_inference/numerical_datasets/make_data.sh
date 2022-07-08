#!/bin/bash
export PYENV_VERSION=PonderNet

inference_step=5
equation_num=10

PYTHONHASHSEED=0 python make_dataset.py \
        --train_data_size=2 \
        --valid_data_size=2 \
        --test_data_size=2 \
        --inference_step=$inference_step \
        --equation_num=$equation_num \
        --output_dir="./data" 

echo "create data"

input_file="depth_1_num_1"
declare -a method=("train" "valid" "test")    # 初期化

for i in {0..2}; do
PYTHONHASHSEED=0 python convert_numerical_data.py \
        --input_file="./data/depth_${inference_step}_num_${equation_num}/${method[$i]}.jsonl" \
        --method=${method[$i]}
echo "convert ${method[$i]}_data"
done