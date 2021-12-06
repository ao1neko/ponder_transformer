#!/bin/bash
export PYENV_VERSION=PonderNet
pyenv versions
conda info -e

WOKE="/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/babi/"

PYTHONHASHSEED=0 python run.py \
    --data_pass=$WOKE/$1.txt \
    --epochs=300 \
    --max_step=6 \
    --batch_size=128 \
    --seed=6 \
    --print_sample_num=20 \
    --valid=true \
    --log_dir=runs/vanilla/step5 \
    --ponder_model=false \
    --device=cuda:2 \
    --train=true 
#--load_pass="best_vanilla_models/1970-01-01 09:00:00_epochs=300,batch_size=128,max_step=6,seed=6,device=cuda:2,beta=1.0,train=true,test=false,valid=true,print_sample_num=3,ponder_model=false_state_dict.pt"

#./run_vanilla.sh task_1
#./run_vanilla.sh task_test_1