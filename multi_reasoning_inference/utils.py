import argparse
from cmath import log
from turtle import pd
import numpy as np
from sklearn.isotonic import check_increasing
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
import json

def read_jsonl_file(input_file:Path):
    with open(input_file, 'r') as input_json_file:
        for json_data in input_json_file:
            if json_data != "\n":
                json_data = json.loads(json_data)
                yield (json_data["input"],json_data["label"])
            
                
def read_jsonl_files(input_file_list:List[Path]): 
    inputs = []
    labels = []
    for input_file in input_file_list:
        for (input, label) in read_jsonl_file(input_file):
            inputs.append(input)
            labels.append(label)
    return (inputs, labels)
        
        

def retrieve_last_string(str:str)-> str:
    match = re.search(r".*answer :\s*(.+?)</s>",str)
    try:
        return match.group(1).replace(' ', '')
    except:
        return None
    
    
def clean_predict_text(predict_text_list:List[str])->List[str]:
    cleaned_predict_text_list = []
    for predict_text in predict_text_list:
        cleaned_text = re.sub(r"<pad>", "", predict_text)
        cleaned_text = re.sub(r"</s>", "", cleaned_text)
        cleaned_predict_text_list.append(cleaned_text)
    return cleaned_predict_text_list


def eliminate_calculated_index(calculated_list:List[int],test_inputs:List[str], test_labels:List[str],inference_acc_list:List[int])-> Tuple[List[str],List[str],List[str]]:
    eliminated_test_inputs=[]
    eliminated_test_labels = []
    eliminated_inference_acc_list = []
    
    for i,(calculated_index,input,label) in enumerate(zip(calculated_list,test_inputs, test_labels)):
        if calculated_index is None:
            eliminated_test_inputs.append(input)
            eliminated_test_labels.append(label)
            eliminated_inference_acc_list.append(inference_acc_list[i])
    return (eliminated_test_inputs, eliminated_test_labels, eliminated_inference_acc_list)



def at_once_check_inference(predict_text,label_text):
    cleaned_predict_text = re.sub(r"<pad>", "", predict_text)
    cleaned_predict_text = re.sub(r"</s>", "", cleaned_predict_text)
    cleaned_label_text = re.sub(r"<pad>", "", label_text)
    cleaned_label_text = re.sub(r"</s>", "", cleaned_label_text)
    
    cleaned_predict_text_list = cleaned_predict_text.split(",")
    cleaned_label_text_list = cleaned_label_text.split(",")
    
    if len(cleaned_label_text_list)!=len(cleaned_predict_text_list): return False
    
    for predict,label in zip(cleaned_predict_text_list,cleaned_label_text_list):
        if predict.replace(' ', '') != label.replace(' ', '') : return False
    return True


def iterative_check_inference(predict_text,label_text,step_index):
    cleaned_predict_text = re.sub(r"<pad>", "", predict_text)
    cleaned_predict_text = re.sub(r"</s>", "", cleaned_predict_text)
    cleaned_label_text = re.sub(r"<pad>", "", label_text)
    cleaned_label_text = re.sub(r"</s>", "", cleaned_label_text)
    
    cleaned_label_text_list = cleaned_label_text.split(",")
    
    if len(cleaned_label_text_list)-1 < step_index : return False
    if cleaned_predict_text.replace(' ', '') != cleaned_label_text_list[step_index].replace(' ', '') : return False
    return True
    
    
def one_token_check_inference(predict_text,label_text,step_index):
    cleaned_predict_text = re.sub(r"<pad>", "", predict_text)
    cleaned_predict_text = re.sub(r"</s>", "", cleaned_predict_text)
    cleaned_label_text = re.sub(r"<pad>", "", label_text)
    cleaned_label_text = re.sub(r"</s>", "", cleaned_label_text)
    
    cleaned_label_text_list = cleaned_label_text.split(",")
    
    if len(cleaned_label_text_list)-1 < step_index : return False
    if cleaned_predict_text.replace(' ', '') != cleaned_label_text_list[step_index].replace(' ', '') : return False
    return True

def at_once_predict(input_id_list,predict_id_list,label_id_list,tokenizer,output_dir)-> int:
    acc_num = 0
    inference_acc_num = 0
    analyze_file_path = Path(output_dir) / Path('analyze.txt')
    err_file_path = Path(output_dir) / Path('analyze_err.txt')
    inference_err_file_path = Path(output_dir) / Path('analyze_inference_err.txt')
    del_analyze_flag = True
    del_analyze_err_flag = True
    del_analyze_inference_err_flag = True
    
    with open(analyze_file_path, 'w') as f, open(err_file_path, 'w') as f_err, open(inference_err_file_path,"w") as f_inference_err:
        for input_ids,predict_ids,label_ids in zip(input_id_list,predict_id_list,label_id_list):
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            predict_text = tokenizer.decode(predict_ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            label_text = tokenizer.decode(label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            #print(f"input:{input_text}")
            #print(f"predict:{predict_text}")
            #print(f"label:{label_text}")
            
            predict = retrieve_last_string(predict_text)
            label =retrieve_last_string(label_text)
            
            inference_true = at_once_check_inference(predict_text,label_text)
            
            if predict is not None and (predict == label): 
                acc_num+=1
                if inference_true:
                    inference_acc_num += 1
                    del_analyze_flag = False
                    f.write(f"input:{input_text}\n")
                    f.write(f"predict:{predict_text}\n")
                    f.write(f"label:{label_text}\n")
                else :
                    del_analyze_inference_err_flag = False
                    f_inference_err.write(f"input:{input_text}\n")
                    f_inference_err.write(f"predict:{predict_text}\n")
                    f_inference_err.write(f"label:{label_text}\n")
            else:
                del_analyze_err_flag = False
                f_err.write(f"input:{input_text}\n")
                f_err.write(f"predict:{predict_text}\n")
                f_err.write(f"label:{label_text}\n")
    if del_analyze_flag : os.remove(analyze_file_path)
    if del_analyze_err_flag : os.remove(err_file_path)
    if del_analyze_inference_err_flag : os.remove(inference_err_file_path)
    return acc_num, inference_acc_num
    
def iterative_predict(input_id_list,predict_id_list,label_id_list,tokenizer,output_dir,inference_acc_list,step_index):
    acc_num = 0
    inference_acc_num = 0
    calculated_list = []
    predict_text_list = []
    analyze_file_path = Path(output_dir) / Path(f'analyze_{step_index}.txt')
    err_file_path = Path(output_dir) / Path(f'analyze_err_{step_index}.txt')
    inference_err_file_path = Path(output_dir) / Path(f'analyze_inference_err_{step_index}.txt')
    del_analyze_flag = True
    del_analyze_err_flag = True
    del_analyze_inference_err_flag = True    
    
    
    with open(analyze_file_path, 'w') as f, open(err_file_path, 'w') as f_err, open(inference_err_file_path,"w") as f_inference_err:
        for index, (input_ids, predict_ids,label_ids) in enumerate(zip(input_id_list,predict_id_list,label_id_list)):
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            predict_text = tokenizer.decode(predict_ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            label_text = tokenizer.decode(label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            predict_text_list.append(predict_text)
            #print(f"input:{input_text}")
            #print(f"predict:{predict_text}")
            #print(f"label:{label_text}")
            
            predict = retrieve_last_string(predict_text)
            calculated_list.append(predict)
            label =retrieve_last_string(label_text)
            
            inference_true = iterative_check_inference(predict_text,label_text,step_index)
            
            if not inference_true : 
                inference_acc_list[index] = 0
            
            if predict is not None and (predict == label): 
                acc_num+=1
                if inference_acc_list[index] == 1:
                    inference_acc_num += 1
                    del_analyze_flag = False
                    f.write(f"input:{input_text}\n")
                    f.write(f"predict:{predict_text}\n")
                    f.write(f"label:{label_text}\n")
                else:
                    del_analyze_inference_err_flag = False
                    f_inference_err.write(f"input:{input_text}\n")
                    f_inference_err.write(f"predict:{predict_text}\n")
                    f_inference_err.write(f"label:{label_text}\n")
            elif predict is not None:
                del_analyze_err_flag = False
                f_err.write(f"input:{input_text}\n")
                f_err.write(f"predict:{predict_text}\n")
                f_err.write(f"label:{label_text}\n")
    if del_analyze_flag : os.remove(analyze_file_path)
    if del_analyze_err_flag : os.remove(err_file_path)
    if del_analyze_inference_err_flag : os.remove(inference_err_file_path)
    
    return acc_num, inference_acc_num, calculated_list, predict_text_list, inference_acc_list

def retrieve_inference(text):
    text = text.replace(' ', '')
    match = re.search(r"</s>.*?</s>(.+?)answer",text)
    try:
        return match.group(1)
    except:
        return None     



def one_token_eliminate_calculated_index(last_token_list:List[int],test_inputs:List[str], test_labels:List[str])-> Tuple[List[str],List[str],List[str]]:
    eliminated_test_inputs=[]
    eliminated_test_labels = []
    
    for last_token_index,input,label in zip(last_token_list,test_inputs, test_labels):
        if not last_token_index:
            eliminated_test_inputs.append(input)
            eliminated_test_labels.append(label)
    return (eliminated_test_inputs, eliminated_test_labels)

def one_token_predict(input_id_list,predict_id_list,label_id_list,tokenizer,output_dir,step_index):
    acc_num = 0
    inference_acc_num = 0
    last_token_list = []
    predict_text_list = []
    analyze_file_path = Path(output_dir) / Path(f'analyze_{step_index}.txt')
    err_file_path = Path(output_dir) / Path(f'analyze_err_{step_index}.txt')
    inference_err_file_path = Path(output_dir) / Path(f'analyze_inference_err_{step_index}.txt')
    
    
    del_analyze_flag = True
    del_analyze_err_flag = True
    del_analyze_inference_err_flag = True
    with open(analyze_file_path, 'w') as f, open(err_file_path, 'w') as f_err, open(inference_err_file_path,"w") as f_inference_err:
        for index, (input_ids, predict_ids,label_ids) in enumerate(zip(input_id_list,predict_id_list,label_id_list)):
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            predict_text = tokenizer.decode(predict_ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            label_text = tokenizer.decode(label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            predict_text_list.append(predict_text)
            
            """
            if step_index % 100 == 0 :
                print(f"input:{input_text}")
                print(f"predict:{predict_text}")
                print(f"label:{label_text}")
            """
            
            bool_last_token = predict_text.startswith("</s>")
            predict = retrieve_last_string(input_text)
            last_token_list.append(bool_last_token)
            label = retrieve_last_string(label_text)
            
            if bool_last_token and (predict == label): 
                acc_num+=1
                predict_inference_str = retrieve_inference(input_text)
                label_inference_str = retrieve_inference("</s></s>"+label_text)
                
                if predict_inference_str == label_inference_str:
                    del_analyze_flag = False
                    inference_acc_num += 1
                    f.write(f"input:{input_text}\n")
                    f.write(f"predict:{predict_text}\n")
                    f.write(f"label:{label_text}\n")
                else:
                    del_analyze_inference_err_flag = False
                    f_inference_err.write(f"input:{input_text}\n")
                    f_inference_err.write(f"predict:{predict_text}\n")
                    f_inference_err.write(f"label:{label_text}\n")
            elif bool_last_token:
                del_analyze_err_flag = False
                f_err.write(f"input:{input_text}\n")
                f_err.write(f"predict:{predict_text}\n")
                f_err.write(f"label:{label_text}\n")
                
    if del_analyze_flag : os.remove(analyze_file_path)
    if del_analyze_err_flag : os.remove(err_file_path)
    if del_analyze_inference_err_flag : os.remove(inference_err_file_path)
    
    return acc_num, inference_acc_num, last_token_list, predict_text_list


