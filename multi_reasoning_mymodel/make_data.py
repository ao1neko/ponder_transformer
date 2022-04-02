from pathlib import Path
import json
from typing import Counter, List,Dict
import argparse
from distutils.util import strtobool
import random
from pathlib import Path
from itertools import count
import random
import itertools
import _jsonnet as jsonnet
import pickle



def pretrain_passage_questuion_generater(config_dict):
    min_value = config_dict["min_value"]
    max_value = config_dict["max_value"]
    mod_num = config_dict["mod_num"]
    
    for a,b in list(itertools.product(range(min_value,max_value+1),range(min_value,max_value+1))):
        new_passage_question={}
        new_passage_question["datas"] = [f"{a} + {b}",f"{(a+b) % mod_num}"] if mod_num else [f"{a} + {b}",f"{a+b}"]
        yield  new_passage_question



     
def pretrain_dataset_generater(save_filename, config_filepath):
    config_dict = json.loads(jsonnet.evaluate_file(str(config_filepath)+".json"))
    save_filename_path = Path(save_filename +"_"+ "_".join([f"{k}:{v}" for k,v in config_dict.items()]) +".json")
    with open(save_filename_path, mode="w") as f:
        f.write("{\n")
        counter = count()
        is_first = True
        
        for passage_questuion in pretrain_passage_questuion_generater(config_dict):
            if not is_first:
                f.write(",\n")
            data_name = f"nfl_{next(counter)}"
            new_data = json.dumps({data_name: passage_questuion}, indent=4)
            new_data = new_data[1:-2]
            f.write(new_data)
            is_first = False
        f.write("\n}\n")
        



###onestep
def onestep_passage_questuion_generater(config_dict):
    seed = config_dict["seed"]
    args_num = config_dict["args_num"]
    args_num = config_dict["args_num"]
    number_file_path = config_dict["number_file_path"]
    mod_num = config_dict["mod_num"]
    
    random.seed(seed)
    number_file_path = Path(number_file_path).expanduser()
    with number_file_path.open(mode="rb") as f:
        number_list = pickle.load(f)
    counter = count()
    for _ in counter:
        random_number_list = random.choices(number_list,k=args_num)
        new_passage_question = {}
        new_passage_question["datas"]= [" + ".join([str(x) for x in random_number_list])]
        while len(random_number_list) > 1:
            random_number_list[0] = (random_number_list[0] + random_number_list.pop(1)) % mod_num if mod_num else (random_number_list[0] + random_number_list.pop(1)) 
            new_passage_question["datas"].append(" + ".join([str(x) for x in random_number_list]))
        yield  new_passage_question

def onestep_dataset_generater(save_filename, config_filepath):
    config_dict = json.loads(jsonnet.evaluate_file(str(config_filepath)))
    max_number_of_question = config_dict["max_number_of_question"]
    passage_set = set()
    
    ignore_config_dict = config_dict.copy()
    del ignore_config_dict["number_file_path"]
    save_filename_path = Path(save_filename +"_"+ "_".join([f"{k}:{v}" for k,v in ignore_config_dict.items()]) +".json")

    with open(save_filename_path, mode="w") as f:
        f.write("{\n")
        counter = count()
        is_first = True
        id = 0
        
        for passage_questuion in onestep_passage_questuion_generater(config_dict):
            passage_string = passage_questuion['datas'][0]
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
            args.config_filepath,
            )
    elif strtobool(args.onestep): 
        onestep_dataset_generater(
            args.output_filepath,
            args.config_filepath,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_filepath",  help="Select output file", type = str)
    parser.add_argument("config_filepath",  help="Select config file", type = str,default=None)
    parser.add_argument("--pretrain",  help="", type = str,default='false')
    parser.add_argument("--onestep",  help="", type = str,default='false')
    
    args = parser.parse_args()
    main(args)
