#!/bin/bash
export PYENV_VERSION=PonderNet

pyenv versions
conda info -e

WOKE01="/work01/aoki0903/PonderNet/multihop_experiment/datas"
method=$1
train_data="arg2_train_100.json"
valid_data="arg2_valid_100.json"
test_data="arg2_test_100.json"

#./run_ponder.sh pretrain
if [ $method = "pretrain" ]; then
    pretrain_data="pretrain_data_10.json"
    PYTHONHASHSEED=0 python main.py \
        --epochs=300 \
        --batch_size=128 \
        --num_layers=6 \
        --seed=6 \
        --emb_dim=512 \
        --beta=0.01 \
        --lambda_p=4 \
        --device=cuda:0 \
        --train=false \
        --pretrain=true \
        --test=false \
        --analyze=false \
        --ponder_model=false \
        --loop_model=true \
        --vanilla_model=false \
        --lr=0.0001 \
        --load_model=false \
        --model_save_path="best_models/loop/pretrain/" \
        --tensorboard_log_dir="tensorboard_log/loop/pretrain/" \
        --json_base_dir=$WOKE01 \
        --train_json_names=$pretrain_data \
        --valid_json_names=$pretrain_data \
        --test_json_names=$pretrain_data \
        --comment="no_comment" \
        --ignore_comment_args train pretrain load_model test analyze ponder_model loop_model vanilla_model valid_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
elif [ $method = "train" ]; then
    PYTHONHASHSEED=0 python main.py \
        --epochs=300 \
        --batch_size=128 \
        --num_layers=6 \
        --seed=6 \
        --emb_dim=512 \
        --beta=0.01 \
        --lambda_p=4 \
        --device=cuda:0 \
        --train=true \
        --pretrain=false \
        --test=false \
        --analyze=false \
        --ponder_model=false \
        --loop_model=true \
        --vanilla_model=false \
        --lr=0.0001 \
        --load_model=true \
        --model_save_path="best_models/loop/train/" \
        --model_load_path="best_models/loop/pretrain/state_dict.pt" \
        --tensorboard_log_dir="tensorboard_log/loop/train/" \
        --json_base_dir=$WOKE01 \
        --train_json_names=$train_data \
        --valid_json_names=$valid_data \
        --test_json_names=$test_data \
        --comment="no_comment" \
        --ignore_comment_args train pretrain load_model test analyze ponder_model loop_model vanilla_model valid_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
elif [ $method = "test" ]; then
    PYTHONHASHSEED=0 python main.py \
        --epochs=300 \
        --batch_size=128 \
        --num_layers=6 \
        --seed=6 \
        --emb_dim=512 \
        --beta=0.01 \
        --lambda_p=4 \
        --device=cuda:0 \
        --train=false \
        --pretrain=false \
        --test=true \
        --analyze=false \
        --ponder_model=false \
        --loop_model=true \
        --vanilla_model=false \
        --lr=0.0001 \
        --load_model=true \
        --model_save_path="best_models/loop/train/" \
        --model_load_path="best_models/loop/train/no_comment:epochs=300,batch_size=128,num_layers=6,seed=6,emb_dim=512,beta=0.01,lambda_p=4,lr=0.0001,train_json_names=['arg2_train_100.json']/state_dict.pt" \
        --json_base_dir=$WOKE01 \
        --train_json_names=$train_data \
        --valid_json_names=$valid_data \
        --test_json_names=$test_data \
        --comment="no_comment" \
        --ignore_comment_args train pretrain load_model test analyze ponder_model loop_model vanilla_model valid_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 

elif [ $method = "analyze" ]; then
    PYTHONHASHSEED=0 python main.py \
        --epochs=300 \
        --batch_size=128 \
        --num_layers=6 \
        --seed=6 \
        --emb_dim=512 \
        --beta=0.01 \
        --lambda_p=4 \
        --device=cuda:0 \
        --train=false \
        --pretrain=false \
        --test=false \
        --analyze=true \
        --ponder_model=false \
        --loop_model=true \
        --vanilla_model=false \
        --lr=0.0001 \
        --load_model=true \
        --model_save_path="best_models/loop/train/" \
        --model_load_path="best_models/loop/train/no_comment:epochs=300,batch_size=128,num_layers=6,seed=6,emb_dim=512,beta=0.01,lambda_p=4,lr=0.0001,train_json_names=['arg2_train_100.json']/state_dict.pt" \
        --json_base_dir=$WOKE01 \
        --train_json_names=$train_data \
        --valid_json_names=$valid_data \
        --test_json_names=$test_data \
        --comment="no_comment" \
        --ignore_comment_args train pretrain load_model test analyze ponder_model loop_model vanilla_model valid_json_names test_json_names comment json_base_dir model_load_path model_save_path tensorboard_log_dir device ignore_comment_args 
else
    echo "invalid args"
    exit 1
fi
