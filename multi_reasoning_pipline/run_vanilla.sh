#!/bin/bash
export PYENV_VERSION=PonderNet

pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment/datas"
method=$1
train_data="arg3_train_10000.json arg4_train_10000.json arg5_train_10000.json"
valid_data="arg5_valid_1000.json"
test_data="arg5_test_1000.json"
pretrain_data="pretrain_data_1000.json"

#./run_vanilla.sh pretrain
if [ $method = "pretrain" ]; then
    PYTHONHASHSEED=0 python main.py \
        --epochs=300 \
        --batch_size=128 \
        --num_layers=4 \
        --seed=6 \
        --emb_dim=512 \
        --device=cuda:1 \
        --pretrain=true \
        --vanilla_model=true \
        --lr=0.0001 \
        --load_model=false \
        --rand_pos_encoder_type=true \
        --model_save_path="best_models/vanilla/pretrain/" \
        --tensorboard_log_dir="tensorboard_log/vanilla/pretrain/" \
        --json_base_dir=$WOKE01 \
        --train_json_names $pretrain_data \
        --valid_json_names $pretrain_data \
        --test_json_names $pretrain_data \
        --comment="no_comment" \
        --ignore_comment_args batch_size epochs beta lambda_p absolute_halting train pretrain load_model test analyze ponder_model loop_model vanilla_model valid_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
elif [ $method = "train" ]; then
    PYTHONHASHSEED=0 python main.py \
        --epochs=1000 \
        --batch_size=128 \
        --num_layers=1 \
        --seed=6 \
        --emb_dim=512 \
        --device=cuda:0 \
        --train=true \
        --vanilla_model=true \
        --lr=0.0001 \
        --load_model=false \
        --model_save_path="best_models/vanilla/train/" \
        --model_load_path="best_models/vanilla/pretrain/no_comment:epochs=300,batch_size=128,num_layers=1,seed=6,emb_dim=512,lr=0.0001,rand_pos_encoder_type=true,train_json_names=['pretrain_data_1000.json']/state_dict.pt" \
        --tensorboard_log_dir="tensorboard_log/vanilla/train/" \
        --json_base_dir=$WOKE01 \
        --train_json_names $train_data \
        --valid_json_names $valid_data \
        --test_json_names $test_data \
        --comment="not_load_model" \
        --ignore_comment_args batch_size epochs beta lambda_p absolute_halting train pretrain load_model test analyze ponder_model loop_model vanilla_model test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
elif [ $method = "test" ]; then
    PYTHONHASHSEED=0 python main.py \
        --batch_size=128 \
        --num_layers=1 \
        --seed=6 \
        --emb_dim=512 \
        --device=cuda:0 \
        --test=true \
        --vanilla_model=true \
        --lr=0.0001 \
        --load_model=true \
        --model_save_path="best_models/vanilla/train/" \
        --model_load_path="best_models/vanilla/train/no_comment:epochs=300,batch_size=128,num_layers=6,seed=6,emb_dim=512,beta=0.01,lambda_p=4,lr=0.0001,train_json_names=['arg2_train_100.json']/state_dict.pt" \
        --json_base_dir=$WOKE01 \
        --train_json_names $train_data \
        --valid_json_names $valid_data \
        --test_json_names $test_data \
        --comment="no_comment" \
        --ignore_comment_args batch_size epochs beta lambda_p absolute_halting train pretrain load_model test analyze ponder_model loop_model vanilla_model valid_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 

elif [ $method = "analyze" ]; then
    PYTHONHASHSEED=0 python main.py \
        --num_layers=1 \
        --seed=6 \
        --emb_dim=512 \
        --device=cuda:0 \
        --analyze=true \
        --vanilla_model=true \
        --lr=0.0001 \
        --load_model=true \
        --model_save_path="best_models/vanilla/train/" \
        --model_load_path="best_models/vanilla/train/no_comment:epochs=1000,batch_size=128,num_layers=1,seed=6,emb_dim=512,lr=0.0001,rand_pos_encoder_type=true,train_json_names=['arg3_train_10000.json', 'arg4_train_10000.json', 'arg5_train_10000.json'],valid_json_names=['arg3_valid_1000.json']/state_dict.pt" \
        --json_base_dir=$WOKE01 \
        --train_json_names $test_data \
        --valid_json_names $test_data \
        --test_json_names $pretrain_data \
        --comment="no_comment" \
        --ignore_comment_args batch_size epochs beta lambda_p absolute_halting train pretrain load_model test analyze ponder_model loop_model vanilla_model valid_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
else
    echo "invalid args"
    exit 1
fi

#        --model_load_path="best_models/vanilla/pretrain/no_comment:epochs=1000,batch_size=128,num_layers=1,seed=6,emb_dim=512,lr=0.0001,rand_pos_encoder_type=true,train_json_names=['arg3_train_10000.json', 'arg4_train_10000.json', 'arg5_train_10000.json'],valid_json_names=['arg3_valid_1000.json']/state_dict.pt" \
