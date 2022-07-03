import argparse
from cmath import e, log
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
from torch.utils.data.dataset import Subset
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List, Dict, Set
import string
import copy
import json

from numerical_data_class import DataInstance,Equation


def main(args):
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    with open(input_file, 'r') as input_json_file, open(output_file, 'w') as output_json_file:
        for json_data in input_json_file:
            if json_data != "\n":
                data_instance = DataInstance().from_json(json.loads(json_data))
                converted_json_data = dict()
                data_instance
                
                output_json_file.write(json.dumps((converted_json_data)))
                output_json_file.write("\n")
            
            
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--input_file', default="./data/depth_1_num_1_train.jsonal", type=str)
    parser.add_argument('--output_file', default="./data/depth_1_num_1_train_dataset.jsonal", type=str)
    args = parser.parse_args()
    main(args)