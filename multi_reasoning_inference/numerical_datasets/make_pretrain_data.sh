#!/bin/bash
export PYENV_VERSION=PonderNet

inference_step=5
equation_num=10

PYTHONHASHSEED=0 python make_pretrain_data.py \
        --seed=10 

echo "create pretrain data"

cp ./data/pretrain/pretrain.jsonl ./data/pretrain/pretrain_test.jsonl
PYTHONHASHSEED=0 python convert_numerical_data.py \
        --input_file="./data/pretrain/pretrain.jsonl" \
        --method="train"
PYTHONHASHSEED=0 python convert_numerical_data.py \
        --input_file="./data/pretrain/pretrain_test.jsonl" \
        --method="test"
echo "convert pretrain_data"

mv ./data/pretrain/pretrain_at_once_forward.jsonl ./data/pretrain/pretrain_at_once.jsonl
mv ./data/pretrain/pretrain_iterative_forward.jsonl ./data/pretrain/pretrain_iterative.jsonl
mv ./data/pretrain/pretrain_test_iterative_forward.jsonl ./data/pretrain/pretrain_test_iterative.jsonl
mv ./data/pretrain/pretrain_one_token_forward.jsonl ./data/pretrain/pretrain_one_token.jsonl
mv ./data/pretrain/pretrain_test_one_token_forward.jsonl ./data/pretrain/pretrain_test_one_token.jsonl
rm ./data/pretrain/pretrain_at_once_backward.jsonl ./data/pretrain/pretrain_at_once_forward_backtrack.jsonl ./data/pretrain/pretrain_iterative_backward.jsonl  ./data/pretrain/pretrain_iterative_forward_backtrack.jsonl ./data/pretrain/pretrain_test_at_once_forward.jsonl ./data/pretrain/pretrain_test_at_once_forward_backtrack.jsonl ./data/pretrain/pretrain_test_at_once_backward.jsonl  ./data/pretrain/pretrain_test_ansonly.jsonl ./data/pretrain/pretrain_test_iterative_forward_backtrack.jsonl ./data/pretrain/pretrain_test_iterative_backward.jsonl ./data/pretrain/pretrain_test_one_token_forward_backtrack.jsonl ./data/pretrain/pretrain_test_one_token_backward.jsonl ./data/pretrain/pretrain_one_token_forward_backtrack.jsonl ./data/pretrain/pretrain_one_token_backward.jsonl 
rm ./data/pretrain/pretrain_at_once_forward_05.jsonl ./data/pretrain/pretrain_test_at_once_forward_09.jsonl ./data/pretrain/pretrain_test_iterative_forward_05.jsonl ./data/pretrain/pretrain_test_iterative_forward_09.jsonl ./data/pretrain/pretrain_iterative_forward_05.jsonl ./data/pretrain/pretrain_iterative_forward_09.jsonl

echo "rmove old data"