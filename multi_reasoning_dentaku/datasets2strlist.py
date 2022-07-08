import re
import argparse
from cmath import log
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List, Dict,Tuple
from pathlib import Path
import json

    
def sample_datasets2strlist()-> Tuple[List[str],List[str]]:
    inputs = ["A = 1 , B = 12 , C = A + B , C = ?", "1", "21"]
    labels = ["A = 1 , B = 12 , C = A + B , C = ?", "1", "21"]
    return (inputs,labels)
    
    
def proofs_to_texts(proofs:str)-> List[str]:
    proofs = re.sub(r".+failure = ", "", proofs)
    proofs = re.sub(r"[\(\)\-\>\]\<]", "", proofs)
    proofs = re.sub(r"%", " ", proofs)
    proofs = re.sub(r"\s+", " ", proofs)
    proof_indexies = proofs.split(" ")
    proof_indexies = [re.sub(r".*?%", "", s) for s in proof_indexies]
    return proof_indexies

def extract_int(proof_indexies:List[str]) -> List[str]:
    extracted_proof_indexies = []
    for proof_index in proof_indexies:
        if "int" in proof_index : extracted_proof_indexies.append(proof_index)
    return extracted_proof_indexies

def proofwriter_datasets2strlist(dataset_path:str)-> Tuple[List[str],List[str]]:    
    with open(Path(dataset_path), 'r') as json_file:
        json_list = list(json_file)
        for str_data in json_list:
            json_data = json.loads(str_data)
            theory = json_data["theory"]
            questions = json_data["questions"]
            triples = json_data["triples"]
            rules = json_data["rules"]
            proof_texts_dict = {}
            proof_texts_dict["FAIL"] = "fail"

            for k,v in triples.items():
                proof_texts_dict[k] = v["text"]
            for k,v in rules.items():
                proof_texts_dict[k] = v["text"]                
                        
            for question_data in questions.values():
                question = " question : " + question_data["question"]
                answer = " answer : " + str(question_data["answer"])
                proofs = question_data["proofsWithIntermediates"][0]["representation"] if "proofsWithIntermediates" in question_data else question_data["proofs"]
                intermediates = question_data["proofsWithIntermediates"][0]["intermediates"] if "proofsWithIntermediates" in question_data else []
                proof_indexies = proofs_to_texts(proofs)
                
                #intのみ使う
                proof_indexies =  extract_int(proof_indexies)
                
                if len(intermediates) > 0:
                    for k,v in intermediates.items():
                        proof_texts_dict[k] = v["text"]
                proof_texts = [proof_texts_dict[i] for i in proof_indexies]
                yield (theory,question,answer,proof_texts)
    
def proofwriter_datasets2strlist_at_once(dataset_path:str)-> Tuple[List[str],List[str]]:
    inputs = []
    labels = []
    for theory,question,answer,proof_texts in proofwriter_datasets2strlist(dataset_path):
        input = theory + " </s> " + question
        inputs.append(input)
        label = " ".join(proof_texts+[answer])
        labels.append(label)
    return (inputs, labels)

def proofwriter_datasets2strlist_iterative(dataset_path:str,test:bool=False)-> Tuple[List[str],List[str]]:
    if not test:
        inputs = []
        labels = []
        for theory,question,answer,proof_texts in proofwriter_datasets2strlist(dataset_path):
            input = theory
            for text in proof_texts+[answer]:
                inputs.append(input + " </s> " + question)
                labels.append(text)
                input += " " + text
        return (inputs, labels)
    else:
        theories = []
        questions = []
        labels = []
        
        for theory,question,answer,proof_texts in proofwriter_datasets2strlist(dataset_path):    
            theories.append(theory)
            questions.append(question)
            labels.append(answer)
        return (theories,questions ,labels)
    

def proofwriter_datasets2strlist_ansonly(dataset_path:str)-> Tuple[List[str],List[str]]:
    inputs = []
    labels = []
    for theory,question,answer,proof_texts in proofwriter_datasets2strlist(dataset_path):
        input = theory + " </s> " + question
        inputs.append(input)
        labels.append(answer)
    return (inputs, labels)