import argparse
from cmath import log
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
from torch.utils.data.dataset import Subset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from typing import List, Dict,Tuple
import re


def retrieve_last_string(str:str)-> str:
    match = re.search(r".*answer : (.+?)</s>",str)
    try:
        return strtobool(match.group(1))
    except:
        return None
    
    
def clean_predict_text(predict_text_list:List[str])->List[str]:
    cleaned_predict_text_list = []
    for predict_text in predict_text_list:
        cleaned_text = re.sub(r"<pad>", "", predict_text)
        cleaned_text = re.sub(r"</s>", "", cleaned_text)
        cleaned_predict_text_list.append(cleaned_text)
    return cleaned_predict_text_list


def eliminate_calculated_index(calculated_list:List[int],test_theories:List[str], test_questions:List[str], test_labels:List[str])-> Tuple[List[str],List[str],List[str]]:
    eliminated_test_theories=[]
    eliminated_test_questions=[] 
    eliminated_test_labels = []
    
    for calculated_index,theory,question,label in zip(calculated_list,test_theories, test_questions, test_labels):
        if calculated_index is None:
            eliminated_test_theories.append(theory)
            eliminated_test_questions.append(question)
            eliminated_test_labels.append(label)
    return (eliminated_test_theories, eliminated_test_questions, eliminated_test_labels)

def proofwriter_at_once_predict(input_id_list,predict_id_list,label_id_list,tokenizer,output_dir)-> int:
    acc_num = 0
    analyze_file_path = Path(output_dir) / Path('analyze.txt')
    err_file_path = Path(output_dir) / Path('analyze_err.txt')
    
    with open(analyze_file_path, 'w') as f, open(err_file_path, 'w') as f_err:
        for input_ids,predict_ids,label_ids in zip(input_id_list,predict_id_list,label_id_list):
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            predict_text = tokenizer.decode(predict_ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            label_text = tokenizer.decode(label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            print(f"input:{input_text}")
            print(f"predict:{predict_text}")
            print(f"label:{label_text}")
            
            predict_bool = retrieve_last_string(predict_text)
            label_bool =retrieve_last_string(label_text)
            if predict_bool is not None and (predict_bool == label_bool): 
                acc_num+=1
                f.write(f"input:{input_text}\n")
                f.write(f"predict:{predict_text}\n")
                f.write(f"label:{label_text}\n")
            else:
                f_err.write(f"input:{input_text}\n")
                f_err.write(f"predict:{predict_text}\n")
                f_err.write(f"label:{label_text}\n")
    return acc_num
    
def proofwriter_iterative_predict(input_id_list,predict_id_list,label_id_list,tokenizer,output_dir,step_index):
    acc_num = 0
    calculated_list = []
    predict_text_list = []
    analyze_file_path = Path(output_dir) / Path(f'analyze_{step_index}.txt')
    err_file_path = Path(output_dir) / Path(f'analyze_err_{step_index}.txt')
    with open(analyze_file_path, 'w') as f, open(err_file_path, 'w') as f_err:
        for input_ids, predict_ids,label_ids in zip(input_id_list,predict_id_list,label_id_list):
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            predict_text = tokenizer.decode(predict_ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            label_text = tokenizer.decode(label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            predict_text_list.append(predict_text)
            print(f"input:{input_text}")
            print(f"predict:{predict_text}")
            print(f"label:{label_text}")
            
            predict_bool = retrieve_last_string(predict_text)
            calculated_list.append(predict_bool)
            
            label_bool =retrieve_last_string(label_text)
            if predict_bool is not None and (predict_bool == label_bool): 
                acc_num+=1
                f.write(f"input:{input_text}\n")
                f.write(f"predict:{predict_text}\n")
                f.write(f"label:{label_text}\n")
            else:
                f_err.write(f"input:{input_text}\n")
                f_err.write(f"predict:{predict_text}\n")
                f_err.write(f"label:{label_text}\n")
    return acc_num, calculated_list, predict_text_list