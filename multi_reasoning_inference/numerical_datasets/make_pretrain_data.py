import argparse
from cmath import e, log
from operator import mod
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




def make_add_data(output_file,useable_char_set: Set = set(string.ascii_uppercase)):
    with open(Path(output_file), 'w') as json_file:
        min_num = 0
        max_num = 99
        mod_num = 100
        for i in range(min_num,max_num+1):
            for j in range(min_num,max_num+1):
                question = random.choice(list(useable_char_set))
                
                equation = Equation()
                equation.left_side = [question] 
                equation.right_side = [i,j]
                equation.calc_char_set()
                
                inference_equation = Equation()
                inference_equation.left_side = [question]
                inference_equation.right_side = [(i + j) % mod_num]
                inference_equation.calc_char_set()
                
                data_instance = DataInstance(useable_char_set=useable_char_set)
                data_instance.equations.append(equation)
                data_instance.forward_inference_equations.append(inference_equation)
                data_instance.forward_backtrack_inference_equations.append(inference_equation)
                data_instance.backward_inference_equations.append(inference_equation)
                data_instance.question = question
                data_instance.answer = (i + j) % mod_num
                data_instance.char_set = set(question)
    
                json_file.write(json.dumps((data_instance.to_json())))
                json_file.write("\n")

def convert_pretrain_data(input_file,output_file):
    with open(output_file, 'w') as output_json_file, open(input_file, 'r') as input_json_file:
        for json_data in input_json_file:
            if json_data != "\n":
                data_instance = DataInstance()
                data_instance.from_json(json_data)
                (equations_str,question_str,answer_str,forward_inference_equations_str,forward_backtrack_inference_equations_str,backward_inference_equations_str) = data_instance.to_str()
                input = equations_str + " </s> " + question_str
                label = forward_inference_equations_str + " , answer : " + answer_str
                converted_json_data = {
                    "input":input, 
                    "label":label
                }
                output_json_file.write(json.dumps((converted_json_data)))
                output_json_file.write("\n")




def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(f"./data/pretrain/")
    temp_file = output_dir / Path(f"add_temp_pretrain_data.jsonl")
    output_file = output_dir / Path(f"add_pretrain_data.jsonl")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    make_add_data(output_file=temp_file)
    convert_pretrain_data(input_file=temp_file, output_file=output_file)
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()
    main(args)
