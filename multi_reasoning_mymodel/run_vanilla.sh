#!/bin/bash
export PYENV_VERSION=PonderNet

pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment/datas"
method=$1
train_data="pretrain_data_max_value:99_min_value:0_mod_num:100.json onestep_train_args_num:10_max_number_of_question:1000_mod_num:100_seed:615.json"
valid_data="onestep_valid_args_num:10_max_number_of_question:1000_mod_num:100_seed:615.json"
test_data="onestep_test_args_num:10_max_number_of_question:1000_mod_num:100_seed:615.json"
pretrain_data="pretrain_data_max_value:99_min_value:0_mod_num:100.json"

#./run_vanilla.sh pretrain
if [ $method = "pretrain" ]; then
    PYTHONHASHSEED=0 python main.py \
        --epochs=500 \
        --batch_size=128 \
        --num_layers=3 \
        --seed=6 \
        --emb_dim=512 \
        --device=cuda:2 \
        --pretrain=true \
        --vanilla_model=true \
        --decoder_model=false \
        --lr=0.0001 \
        --load_model=false \
        --rand_pos_encoder_type=true \
        --model_save_path="best_models/vanilla/pretrain/" \
        --tensorboard_log_dir="tensorboard_log/vanilla/pretrain/" \
        --json_base_dir=$WOKE01 \
        --train_json_names $pretrain_data \
        --valid_json_names $pretrain_data \
        --test_json_names $pretrain_data \
        --comment="" \
        --ignore_comment_args batch_size epochs train pretrain load_model test analyze valid_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
elif [ $method = "train" ]; then
    PYTHONHASHSEED=0 python main.py \
        --epochs=5000 \
        --batch_size=128 \
        --num_layers=3 \
        --seed=6 \
        --emb_dim=512 \
        --device=cuda:1 \
        --train=true \
        --vanilla_model=false \
        --decoder_model=false \
        --decoder_multi_model=true \
        --lr=0.0001 \
        --load_model=false \
        --model_save_path="best_models/vanilla/train/" \
        --model_load_path="best_models/vanilla/train//state_dict.pt" \
        --tensorboard_log_dir="tensorboard_log/vanilla/train/" \
        --json_base_dir=$WOKE01 \
        --train_json_names $train_data \
        --valid_json_names $valid_data \
        --test_json_names $test_data \
        --comment="" \
        --ignore_comment_args decoder_model vanilla_model batch_size epochs train pretrain test_json_names valid_json_names load_model test analyze test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
elif [ $method = "test" ]; then
    PYTHONHASHSEED=0 python main.py \
        --batch_size=128 \
        --num_layers=3 \
        --seed=6 \
        --emb_dim=512 \
        --device=cuda:1 \
        --test=true \
        --vanilla_model=false \
        --decoder_model=false \
        --decoder_multi_model=true \
        --lr=0.0001 \
        --load_model=true \
        --model_save_path="best_models/vanilla/train/" \
        --model_load_path="best_models/vanilla/train/:num_layers=3,seed=6,emb_dim=512,decoder_multi_model=true,lr=0.0001,rand_pos_encoder_type=true,train_json_names=['pretrain_data_max_value:99_min_value:0_mod_num:100.json', 'onestep_train_args_num:10_max_number_of_question:1000_mod_num:100_seed:615.json']/state_dict.pt" \
        --json_base_dir=$WOKE01 \
        --train_json_names $train_data \
        --valid_json_names $valid_data \
        --test_json_names $test_data \
        --comment="" \
        --ignore_comment_args batch_size epochs train pretrain load_model test analyze valid_json_names test_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 

elif [ $method = "analyze" ]; then
    PYTHONHASHSEED=0 python main.py \
        --batch_size=128 \
        --num_layers=3 \
        --seed=6 \
        --emb_dim=512 \
        --device=cuda:1 \
        --analyze=true \
        --vanilla_model=false \
        --decoder_model=false \
        --decoder_multi_model=true \
        --lr=0.0001 \
        --load_model=true \
        --model_save_path="best_models/vanilla/train/" \
        --model_load_path="best_models/vanilla/train/:num_layers=3,seed=6,emb_dim=512,decoder_multi_model=true,lr=0.0001,rand_pos_encoder_type=true,train_json_names=['pretrain_data_max_value:99_min_value:0_mod_num:100.json', 'onestep_train_args_num:10_max_number_of_question:1000_mod_num:100_seed:615.json']/state_dict.pt" \
        --json_base_dir=$WOKE01 \
        --train_json_names $test_data \
        --valid_json_names $test_data \
        --test_json_names $test_data \
        --comment="" \
        --ignore_comment_args batch_size epochs train pretrain load_model test analyze valid_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
else
    echo "invalid args"
    exit 1
fi
# ./run_vanilla.sh train