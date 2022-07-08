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
from typing import List, Dict, Set,Tuple
import string
import copy
import json

from numerical_data_class import DataInstance,Equation




def convert_numerical_datasets_ansonly(data_instance:DataInstance)-> Tuple[List[str],List[str]]:
    (equations_str,question_str,answer_str,forward_inference_equations_str,forward_backtrack_inference_equations_str,backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str
    label = "answer : " + answer_str
    yield (input, label)


def convert_numerical_datasets_at_once(data_instance:DataInstance,inference_type="forward")-> Tuple[List[str],List[str]]:
    (equations_str,question_str,answer_str,forward_inference_equations_str,forward_backtrack_inference_equations_str,backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str
    if inference_type == "forward":
        label = forward_inference_equations_str + " , answer : " + answer_str
    elif inference_type == "forward_backtrack":
        label = forward_backtrack_inference_equations_str + " , answer : " + answer_str
    elif inference_type == "backward":
        label = backward_inference_equations_str + " , answer : " + answer_str
    yield (input, label)
    

def convert_numerical_datasets_iterative(data_instance:DataInstance,inference_type="forward",test:bool = False)-> Tuple[List[str],List[str]]:
    (equations_str,question_str,answer_str,forward_inference_equations_str,forward_backtrack_inference_equations_str,backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str + "</s>"
    if inference_type == "forward":
        inference_equations = data_instance.forward_inference_equations
    elif inference_type == "forward_backtrack":
        inference_equations = data_instance.forward_backtrack_inference_equations
    elif inference_type == "backward":
        inference_equations = data_instance.backward_inference_equations
    inference_equations = [" + ".join([str(arg) for arg in x.left_side]) + " = " + " + ".join([str(arg) for arg in x.right_side]) for x in inference_equations]
    
    
    if not test:
        for equation in inference_equations + ["answer : " + answer_str]:
            input += " , " + equation
            yield (input, equation)
    else:
        input = equations_str + " </s> " + question_str + "</s>"
        if inference_type == "forward":
            label = forward_inference_equations_str + " , answer : " + answer_str
        elif inference_type == "forward_backtrack":
            label = forward_backtrack_inference_equations_str + " , answer : " + answer_str
        elif inference_type == "backward":
            label = backward_inference_equations_str + " , answer : " + answer_str
        yield (input,label)



def main(args):
    input_file = Path(args.input_file)
    input_file_stem = input_file.stem
    input_file_parent = input_file.parent
    
    method = args.method
    test = True if method=="test" else False
    output_file_tail_list = ["ansonly","at_once_forward","at_once_forward_backtrack","at_once_backward","iterative_forward","iterative_forward_backtrack","iterative_backward"]
    
    
    for index,file_tail in enumerate(output_file_tail_list):
        output_file = input_file_parent / (input_file_stem + "_" + file_tail + ".jsonl")
        with open(output_file, 'w') as output_json_file, open(input_file, 'r') as input_json_file:
            for json_data in input_json_file:
                if json_data != "\n":
                    data_instance = DataInstance()
                    data_instance.from_json(json_data)
                    generator = [
                        convert_numerical_datasets_ansonly(data_instance),
                        convert_numerical_datasets_at_once(data_instance,inference_type="forward"),
                        convert_numerical_datasets_at_once(data_instance,inference_type="forward_backtrack"),
                        convert_numerical_datasets_at_once(data_instance,inference_type="backward"),
                        convert_numerical_datasets_iterative(data_instance,inference_type="forward",test=test),
                        convert_numerical_datasets_iterative(data_instance,inference_type="forward_backtrack",test=test),
                        convert_numerical_datasets_iterative(data_instance,inference_type="backward",test=test),
                    ][index]
                    for input, label in generator:
                        converted_json_data = {
                            "input":input, 
                            "label":label
                        }
                        output_json_file.write(json.dumps((converted_json_data)))
                        output_json_file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--input_file', default="./data/depth_1_num_1_train.jsonal", type=str)
    parser.add_argument('--method', default="train", type=str)
    args = parser.parse_args()
    main(args)