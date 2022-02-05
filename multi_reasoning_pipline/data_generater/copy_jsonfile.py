import json
import random
import itertools
import os
import argparse
from itertools import count
from more_itertools import chunked
from copy import deepcopy
from numerical_data_generation import NumericDataGenarator
import argparse
from pathlib import Path
import pickle
import random
import json


def main(args):
    numerical_config_file_path = args.numerical_config_file_path
    config_file_path = args.config_file_path
    copied_config_file_path = args.copied_config_file_path
    
    with open(config_file_path,"r") as fr:
        json_dic = json.load(fr)
        
    json_dic['number_file_path'] = str(numerical_config_file_path)
    with open(copied_config_file_path, 'w') as fw:
        json.dump(json_dic, fw, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path", help="Select file path", type=Path)
    parser.add_argument("copied_config_file_path", help="Select file path", type=Path)
    parser.add_argument("numerical_config_file_path", help="Select file path", type=Path)
    args = parser.parse_args()
    main(args)
