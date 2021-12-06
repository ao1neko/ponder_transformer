import os
import argparse
import json
import random
from pathlib import Path
from pprint import pprint

from . import edit_functions
from .edit_functions import *
from .numerical_data_generation import NumericDataGenarator

class EquationEditor:
    def __init__(self, config_file_path, numerical_data_generator):

        self.numerical_data_generator = numerical_data_generator
        
        with open(config_file_path, mode="r") as f:
            self.config_dict = json.load(f)
        
        if not (self.config_dict["seed"] == "None"):
            assert type(self.config_dict["seed"]) is int, "Random seed is not int!"
            self.random_module = random.Random(self.config_dict["seed"])
        else:
            self.random_module = random
        
        assert os.environ.get('PYTHONHASHSEED') == "0", "Set enviroment variable \"PYTHONHASHSEED\" = \"0\""    
            
        self.edit_functions_dict = {f: globals()[f] for f in edit_functions.__all__}

        self.probabilities = []
        self.functions = []
        self.indexes = list(range(len(self.config_dict["edit_rules"])))

        for d in self.config_dict["edit_rules"]:
            self.probabilities.append(float(d["probability"]))
            self.functions.append(self.edit_functions_dict[d["function"]])

    
    def __call__(self, operator_config, assignment_configs):
        
        while True:
            index = self.random_module.choices(self.indexes, weights=self.probabilities)[0]
            yield self.functions[index](
                operator_config,
                assignment_configs,
                self.config_dict["edit_rules"][index],
                self.numerical_data_generator,
                self.random_module
            )
            
        




def main(args):
    question_generator = NumericDataGenarator(args.question_config_file_path)
    editor =  EquationEditor(args.edit_config_file_path, question_generator)

    for i in range(100):
        operator_config, assignment_configs = next(question_generator(generate_config=True))
        pprint(operator_config)
        pprint(assignment_configs)
        print(question_generator.get_pqa_triple_from_configs(operator_config, assignment_configs))
        print("========================================")

        operator_config, assignment_configs = next(editor(operator_config, assignment_configs))
        pprint(operator_config)
        pprint(assignment_configs)
        print(question_generator.get_pqa_triple_from_configs(operator_config, assignment_configs, separate=True))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_config_file_path",  help="Select file path", type=Path, default="configs/config.json")
    parser.add_argument("--edit_config_file_path",  help="Select file path", type=Path, default="configs/edit_config.json")
    args = parser.parse_args()
    main(args)
