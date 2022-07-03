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
    data_size = args.data_size
    output_file = Path(args.output_file)
    maked_data_size = 0
    equations_stack = []
    with open(Path(output_file), 'w') as json_file:
        while data_size > maked_data_size :
            data_instance = DataInstance().make_instance(inference_step=3,equation_num=3)
            equations_str,question_str,answer_str,forward_inference_equations_str,forward_backtrack_inference_equations_str,backward_inference_equations_str = data_instance.to_str()
            if equations_str not in equations_stack :
                equations_stack.append(equations_str)
                maked_data_size += 1
                json_file.write(json.dumps((data_instance.to_json())))
                json_file.write("\n")
            
            
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--data_size', default=100, type=int)
    parser.add_argument('--output_file', default="./data/depth_1_num_1_train.jsonal", type=str)

    args = parser.parse_args()
    main(args)
