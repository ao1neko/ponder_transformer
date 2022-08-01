#!/bin/bash
export PYENV_VERSION=PonderNet

file_list="train_ansonly.jsonl valid_ansonly.jsonl test_ansonly.jsonl test_at_once_backward.jsonl train_at_once_backward.jsonl valid_at_once_backward.jsonl train_at_once_forward_backtrack.jsonl test_at_once_forward_backtrack.jsonl valid_at_once_forward_backtrack.jsonl test_at_once_forward.jsonl train_at_once_forward.jsonl valid_at_once_forward.jsonl test_iterative_backward.jsonl train_iterative_backward.jsonl valid_iterative_backward.jsonl test_iterative_forward_backtrack.jsonl train_iterative_forward_backtrack.jsonl valid_iterative_forward_backtrack.jsonl test_iterative_forward.jsonl train_iterative_forward.jsonl valid_iterative_forward.jsonl train_at_once_dot.jsonl valid_at_once_dot.jsonl test_at_once_dot.jsonl train_at_once_forward_05.jsonl valid_at_once_forward_05.jsonl test_at_once_forward_05.jsonl train_at_once_forward_09.jsonl valid_at_once_forward_09.jsonl test_at_once_forward_09.jsonl train_iterative_dot.jsonl valid_iterative_dot.jsonl test_iterative_dot.jsonl  train_iterative_forward_05.jsonl valid_iterative_forward_05.jsonl test_iterative_forward_05.jsonl train_iterative_forward_09.jsonl valid_iterative_forward_09.jsonl test_iterative_forward_09.jsonl train_one_token_forward.jsonl valid_one_token_forward.jsonl test_one_token_forward.jsonl train_one_token_forward_backtrack.jsonl valid_one_token_forward_backtrack.jsonl test_one_token_forward_backtrack.jsonl train_one_token_backward.jsonl valid_one_token_backward.jsonl test_one_token_backward.jsonl"
concat_file_list="depth_1_distractor_3 depth_2_distractor_3 depth_3_distractor_3 depth_4_distractor_3 depth_5_distractor_3"
output_file="./data/depth_1_5_distractor_3"
pretrain_dir="./data/pretrain"
mkdir -p $output_file

make_str_file_list () {
    string=""
    for concat_file in $concat_file_list; do
        old_string=$string
        string+="./data/${concat_file}/$1 "
    done
    echo $string
}

for file in $file_list; do
    str_file_list=`make_str_file_list ${file}`
    cat $str_file_list > $output_file/$file
    echo "creat $output_file${file}"
done

cat $pretrain_dir/pretrain_ansonly.jsonl >> $output_file/train_ansonly.jsonl
cat $pretrain_dir/pretrain_at_once.jsonl >> $output_file/train_at_once_forward.jsonl
cat $pretrain_dir/pretrain_at_once.jsonl >> $output_file/train_at_once_backward.jsonl
cat $pretrain_dir/pretrain_at_once.jsonl >> $output_file/train_at_once_forward_backtrack.jsonl
cat $pretrain_dir/pretrain_iterative.jsonl >> $output_file/train_iterative_forward.jsonl
cat $pretrain_dir/pretrain_iterative.jsonl >> $output_file/train_iterative_backward.jsonl
cat $pretrain_dir/pretrain_iterative.jsonl >> $output_file/train_iterative_forward_backtrack.jsonl
cat $pretrain_dir/pretrain_one_token.jsonl >> $output_file/train_one_token_forward.jsonl
cat $pretrain_dir/pretrain_one_token.jsonl >> $output_file/train_one_token_forward_backtrack.jsonl
cat $pretrain_dir/pretrain_one_token.jsonl >> $output_file/train_one_token_backward.jsonl

echo "add pretrain data"