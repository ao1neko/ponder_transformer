#!/bin/bash
rm /work01/aoki0903/genbert_mini -r -f


export PYENV_VERSION=genbertn
pyenv versions


python -m genbert.datasets.my_drop configs/drop_my_dataset.jsonnet \
    --input-file datasets/$1 \
    --output-file datasets/drop_dataset/mini_drop_dataset_dev.pickle

env seed=42 \
    train_data_path=datasets/drop_dataset/mini_drop_dataset_dev.pickle \
    validation_data_path=datasets/drop_dataset/mini_drop_dataset_dev.pickle \
    devices="0" \
    pretrained_weights="" \
    allennlp train \
        --serialization-dir /work01/aoki0903/genbert_mini \
        --include-package genbert configs/genbert.jsonnet

mkdir /work01/aoki0903/genbert_mini/results

allennlp predict \
    --include-package genbert \
    --cuda-device 0 \
    --use-dataset-reader \
    --output-file /work01/aoki0903/genbert_mini/results/output.jsonl \
    /work01/aoki0903/genbert_mini/model.tar.gz \
    datasets/drop_dataset/mini_drop_dataset_dev.pickle