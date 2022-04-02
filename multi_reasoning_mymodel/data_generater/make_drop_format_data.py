import json
from pathlib import Path
import pickle

import random
import itertools
import os
import argparse
from itertools import count
from more_itertools import chunked
from copy import deepcopy
from numerical_data_generation import NumericDataGenarator
from distutils.util import strtobool
import _jsonnet as jsonnet


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True



def read_template(filename):
    with open(filename, mode="r") as f:
        json_dict = json.load(f)
    return json_dict



def passage_questuion_generater(config_filepath):
    pwd = os.path.dirname(__file__)
    
    template_dict = read_template(os.path.join(pwd, "template/drop_template_one_passage.json"))
    qa_template = read_template(os.path.join(pwd, "template/drop_template_question.json"))
    
    numeric_data_ganerator = NumericDataGenarator(config_filepath=config_filepath)

    counter = count()
    for passage, question, answer in numeric_data_ganerator():
        new_passage_question = deepcopy(template_dict)
        new_passage_question["passage"] = passage[4:]

        for q, a in zip(question, answer):
            new_qa = deepcopy(qa_template)
            new_qa["query_id"] = str(next(counter))
            new_qa["question"] = q[2:]

            # 数字であればnumber(生成して解答)に, そうでなければspan(抽出での解答)とする
            if is_num(a):
                new_qa["answer"]["number"] = a
            else:
                new_qa["answer"]["spans"].append(a)
                
            new_passage_question["qa_pairs"].append(new_qa)
            
        
        yield new_passage_question



def pretrain_passage_questuion_generater(min_value,max_value):
    pwd = os.path.dirname(__file__)
    template_dict = read_template(os.path.join(pwd, "template/drop_template_one_passage.json"))
    qa_template = read_template(os.path.join(pwd, "template/drop_template_question.json"))
    
    counter = count()
    question = "="
    
    for a,b in list(itertools.product(range(min_value,max_value+1),range(min_value,max_value+1))):
        passage = 1
        answer = 1
        new_passage_question = deepcopy(template_dict)
        new_passage_question["passage"] = f"{a} + {b}"

        new_qa = deepcopy(qa_template)
        new_qa["query_id"] = str(next(counter))
        new_qa["question"] = question
        new_qa["answer"]["number"] = f"{a+b}"
        
        new_passage_question["qa_pairs"].append(new_qa)
            
        
        yield new_passage_question    



def drop_dataset_generater(save_filename, config_filepath,max_number_of_question):
    passage_set = set()
    
    with open(save_filename, mode="w") as f:
        f.write("{\n")
        counter = count()
        is_first = True
        id = 0
        
        for passage_questuion in passage_questuion_generater(config_filepath):
            passage_string = passage_questuion['passage']
            if passage_string  not in passage_set:
                passage_set.add(passage_string)
                if not is_first:
                    f.write(",\n")
                id = next(counter)
                data_name = f"nfl_{id}"
                new_data = json.dumps({data_name: passage_questuion}, indent=4)
                new_data = new_data[1:-2]
                
                f.write(new_data)
                is_first = False
            if id == max_number_of_question-1: break
        f.write("\n}\n")
        
        
        
def pretrain_dataset_generater(save_filename, max_value):
    with open(save_filename, mode="w") as f:
        f.write("{\n")
        counter = count()
        is_first = True
        
        for passage_questuion in pretrain_passage_questuion_generater(min_value=0,max_value=max_value):
            if not is_first:
                f.write(",\n")
            data_name = f"nfl_{next(counter)}"
            new_data = json.dumps({data_name: passage_questuion}, indent=4)
            new_data = new_data[1:-2]
            f.write(new_data)
            is_first = False
        f.write("\n}\n")




###onestep
def onestep_passage_questuion_generater(config_filepath):
    pwd = os.path.dirname(__file__)
    template_dict = read_template(os.path.join(pwd, "template/drop_template_one_passage.json"))
    qa_template = read_template(os.path.join(pwd, "template/drop_template_question.json"))
    
    config_dict = json.loads(jsonnet.evaluate_file(str(config_filepath)))
    random.seed(config_dict["seed"])
    args_num = config_dict["args_num"]
    number_file_path = config_dict.get("number_file_path")
    number_file_path = Path(number_file_path).expanduser()
    with number_file_path.open(mode="rb") as f:
        number_list = pickle.load(f)


    counter = count()
    question = "="
    
    for i in counter:
        random_number_list = random.choices(number_list,k=args_num)
        new_passage_question = deepcopy(template_dict)
        new_passage_question["passage"] = " + ".join(random_number_list)
        
        new_qa = deepcopy(qa_template)
        new_qa["query_id"] = str(i)
        new_qa["question"] = question
        
        answer_list = []
        while len(random_number_list) > 1:
            random_number_list[0] += random_number_list.pop(1)
            answer_list.append(" + ".join(random_number_list))
        new_qa["answer"]["number"] = answer_list
        new_passage_question["qa_pairs"].append(new_qa)
        yield new_passage_question  

def onestep_dataset_generater(save_filename, config_filepath,max_number_of_question):
    passage_set = set()
    with open(save_filename, mode="w") as f:
        f.write("{\n")
        counter = count()
        is_first = True
        id = 0
        
        for passage_questuion in onestep_passage_questuion_generater(config_filepath):
            passage_string = passage_questuion['passage']
            if passage_string  not in passage_set:
                passage_set.add(passage_string)
                if not is_first:
                    f.write(",\n")
                id = next(counter)
                data_name = f"nfl_{id}"
                new_data = json.dumps({data_name: passage_questuion}, indent=4)
                new_data = new_data[1:-2]
                f.write(new_data)
                is_first = False
            if id == max_number_of_question-1: break
        f.write("\n}\n")




def main(args):
    if strtobool(args.pretrain):
        pretrain_dataset_generater(
            args.output_filepath,
            args.pretrain_max_value
            )
    elif strtobool(args.onestep): 
        onestep_dataset_generater(
            args.output_filepath,
            args.config_filepath, 
            args.max_number_of_question
            )
    else:
        drop_dataset_generater(
            args.output_filepath,
            args.config_filepath, 
            args.max_number_of_question
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_filepath",  help="Select output file", type = str)
    parser.add_argument("--config_filepath",  help="Select config file", type = str,default=None)
    parser.add_argument("--max_number_of_question",  help="max_number_of_question", type = int,default=100)
    parser.add_argument("--pretrain",  help="", type = str,default='false')
    parser.add_argument("--onestep",  help="", type = str,default='false')
    parser.add_argument("--pretrain_max_value",  help="max_number", type = int,default=100)
    
    args = parser.parse_args()
    main(args)
