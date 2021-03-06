import os
import json
import itertools
import string
import argparse
from copy import deepcopy
from itertools import permutations, chain, count
from pprint import pprint
import random
import operators

from operators import *
from numerical_question import NumericQuestion





class NumericDataGenarator:
    def __init__(self, config_filepath):
        with open(config_filepath, mode="r") as f:
            self.config_dict = json.load(f)

        if not (self.config_dict["seed"] == "None"):
            assert type(self.config_dict["seed"]) is int, "Random seed is not int!"
            self.random_module = random.Random(self.config_dict["seed"])
        else:
            self.random_module = random

        
            
        
        assert os.environ.get('PYTHONHASHSEED') == "0", "Set enviroment variable \"PYTHONHASHSEED\" = \"0\""
        
        self.number_of_symbols = self.config_dict["number_of_symbols"]
        assert 2 <= self.number_of_symbols <= 26, "number_of_symbols is out of range"

        #self.symbols = (string.ascii_lowercase + string.ascii_uppercase)[:self.number_of_symbols]
        self.symbols = string.ascii_lowercase[:self.number_of_symbols]
        
        self.max_number_of_question = self.config_dict["max_number_of_question"]
        
        
        self.max_value = self.config_dict["max_value"]
        self.min_value = self.config_dict["min_value"]
        self.generation_rules = self.config_dict["generation_rules"]
        
        
        self.operators_dict = {o: globals()[o] for o in operators.__all__} 
        
        self.output_type = self.config_dict["output_type"]
        self.dtype = self.config_dict["dtype"]

        if self.dtype=="int":
            self.random_func = self.random_module.randint
        elif self.dtype=="float":
            self.random_func = lambda a, b : round(self.random_module.uniform(a, b), 2)
        else:
            raise NotImplementedError(f"Dtype \"{self.dtype}\" in config file is not defined")


        
    def format_assert(self, format_for_assertion, operator_acceptable_formats):
        """
        ??????????????????, ??????, ????????????????????????????????????
        """
        
        if ("*num_var" in operator_acceptable_formats) or ("*var_num" in operator_acceptable_formats):
            return True
        
        all_num = all(map(lambda x: x == "num", format_for_assertion))
        all_var = all(map(lambda x: x == "var", format_for_assertion))
        
        
        for acceptable_format in operator_acceptable_formats:
            if type(acceptable_format) is tuple:
                
                if acceptable_format == format_for_assertion:
                    return True
                
            elif type(acceptable_format) is str:
                if acceptable_format.startswith("*num_var") or acceptable_format.startswith("*var_num"):
                    n = int(acceptable_format.split(":")[-1])
                    if len(format_for_assertion) >= n:
                        return True
                
                elif acceptable_format.startswith("*num") and all_num:
                    n = int(acceptable_format.split(":")[-1])
                    if len(format_for_assertion) >= n:
                        return True

                elif acceptable_format.startswith("*var") and all_var:
                    n = int(acceptable_format.split(":")[-1])
                    if len(format_for_assertion) >= n:
                        return True

                else:
                    tuple_acceptable_format = (acceptable_format, )
                    if tuple_acceptable_format == format_for_assertion:
                        return True        
                    
            else:
                raise Exception(f"type : {type(acceptable_format)} can not be handled")
        

        return False

        
        
    def instantiate_format(self, possible_formats, assignment_format_type, temp_assignment_configs, commutative, generation_type, assumption_length=None):

        # (generation_type == "template") =>(?????????) (assumption_length is not  None)
        assert (not generation_type == "template") or (assumption_length is not  None), "Input \"assumption_length\" when you use \"generation\""
        
        if len(possible_formats) == 0:
            return []
        
        #?????????????????????????????????
        possible_formats = list(map(lambda x: tuple(x) if type(x) is list else x, possible_formats))
        
        if commutative:
            #possible_formats += list(permutations(possible_formats))
            temp_possible_formats = []
            for pf in possible_formats:
                if not(type(pf) is tuple):
                    continue
                temp_possible_formats += list(map(tuple, permutations(pf)))

            # ??????????????????????????????
            possible_formats = list(set(possible_formats + temp_possible_formats))
        
        possible_formats = sorted(possible_formats, key=hash)
        
        temp_assignment_config_variables = [tac["variable"] for tac in temp_assignment_configs]
        selected_format = self.random_module.choice(possible_formats)
        instantiated_format = []
        
        if type(selected_format) is tuple:

            # ??????????????????, ??????, ????????????????????????????????????
            format_for_assertion = tuple(map(lambda x : "var" if type(x) is int else x, selected_format))
            
            operator_acceptable_formats = self.operators_dict[assignment_format_type].arg_formats
            
            
            assert self.format_assert(format_for_assertion, operator_acceptable_formats), f"\"{assignment_format_type}\" is not support \"{format_for_assertion}\"."

            
            for elem in selected_format:
                if elem == "num":
                    instantiated_format.append(str(self.random_func(self.min_value, self.max_value)))
                elif elem == "var":
                    instantiated_format.append(self.random_module.choices(temp_assignment_config_variables)[0])
                elif type(elem) is int:
                    assert generation_type=="template", "generation_rules:type = \"random\" can not use index selection."
                    # - ?????????????????????????????????, temp_assignment_config_variables(?????????)?????????????????????????????????????????????????????????. (temp_assignment_config_variables????????????????????????????????????????????????????????????????????????????????????)
                    if elem < 0:
                        elem = assumption_length + elem
                    
                    instantiated_format.append(temp_assignment_config_variables[elem])
                else:
                    raise NotImplementedError()
        
        
        else:

            # ??????????????????, ??????, ????????????????????????????????????
            format_for_assertion = "var" if type(selected_format) is int else selected_format

            
            operator_acceptable_formats = self.operators_dict[assignment_format_type].arg_formats
            
            assert self.format_assert((format_for_assertion, ), operator_acceptable_formats), f"\"{assignment_format_type}\" is not support \"{format_for_assertion}\"."
            
            if selected_format == "num":
                instantiated_format.append(str(self.random_func(self.min_value, self.max_value)))
            elif selected_format == "var":
                instantiated_format.append(self.random_module.choices(temp_assignment_config_variables)[0])
            elif type(selected_format) is int:
                assert generation_type=="template", "generation_rules:type = \"random\" can not use index selection."
                if selected_format < 0:
                    elem = assumption_length + selected_format
                
                instantiated_format.append(temp_assignment_config_variables[selected_format])
            else:
                raise NotImplementedError()

                
        return instantiated_format





    
    

    def generator_of_template_configs(self, generation_rule):
        assert generation_rule["type"] == "template", "generation_rule's type is not match."
        
        while True:
            assignment_configs = []
            shuffled_symbol_list = self.random_module.sample(self.symbols, len(self.symbols))
        
            for assignment_format in generation_rule["assignment_format"]:
                temp_symbol = shuffled_symbol_list.pop()
                # None ?????????false?????????
                commutative = bool(assignment_format.get("commutative"))

                if type(assignment_format["type"]) is list:
                    selected_assignment_type = self.random_module.choice(assignment_format["type"])
                else:
                    selected_assignment_type = assignment_format["type"]
                    
                assignment_configs.append(
                    {
                        "variable" : temp_symbol,
                        "type" : selected_assignment_type,
                        "format" : self.instantiate_format(
                            assignment_format["format"],
                            selected_assignment_type,
                            assignment_configs,
                            commutative,
                            "template",
                            assumption_length=len(generation_rule["assignment_format"])
                        )
                    }
                )

                

            commutative = bool(generation_rule["operator"].get("commutative"))

            if type(generation_rule["operator"]["type"]) is list:
                selected_ope = self.random_module.choices(generation_rule["operator"]["type"], weights=generation_rule["operator"]["selection_probabilities"])[0]
            else:
                selected_ope = generation_rule["operator"]["type"]

            
            operator_config = {
                "ope"  : selected_ope,
                "format" : self.instantiate_format(
                    generation_rule["operator"]["format"],
                    selected_ope,
                    assignment_configs,
                    commutative,
                    "template",
                    assumption_length=len(generation_rule["assignment_format"])
                )
            }

            yield operator_config, assignment_configs
            

            

    def get_possible_assignment_format(self, assignment_format_list, generation_step_capacity, number_of_available_variable):
        """
        ??????, ??????????????????????????????????????????????????????????????????????????????. 
        ??????1 :??????????????????????????????????????????????????????
        ??????2 :?????????????????????????????????????????????, ???????????????????????????????????????????????????
        """
        possible_assignment_format = []
        
        for assignment_format in assignment_format_list:
            if assignment_format["step_weight"] > generation_step_capacity:
                continue
            elif len(list(filter(lambda x: x=="var", assignment_format["format"]))) > number_of_available_variable:
                continue
            else:
                possible_assignment_format.append(assignment_format)

        return possible_assignment_format
                
        

        


    def generator_of_random_configs(self, generation_rule):
        assert generation_rule["type"] == "random", "generation_rule's type is not match."
        assert all(map(lambda x: len(x["format"]) == 1, generation_rule["assignment_format"])), "assignment_format of random generation must have \"format\" that length is 1."

        
        min_generation_step = generation_rule["reasning_step"]["min"]
        max_generation_step = generation_rule["reasning_step"]["max"]
        assert min_generation_step, "\"reasning_step:min\" must be longer than 2"

        
        # ?????????????????????????????????????????????????????????????????????, ???????????????(????????????0??????????????????)config????????????. 
        substitution_num_step = next(filter(lambda x: x["type"] == "Substitution" and x["format"] == ["num"], generation_rule["assignment_format"]))["step_weight"]


        # yield????????????
        while True:
            shuffled_symbol_list = self.random_module.sample(self.symbols, len(self.symbols))


            possible_format_list = generation_rule["assignment_format"]
            possible_format_selection_probability = [pf["probability"] for pf in possible_format_list]
            generation_step_capacity = self.random_module.randint(min_generation_step, max_generation_step)

            # ???????????????????????????????????????????????????????????????
            assignment_configs = [
                {
                    "variable" : shuffled_symbol_list.pop(),
                    "type" : "Substitution",
                    "format" : [str(self.random_func(self.min_value, self.max_value))]
                }
            ]         
            generation_step_capacity -= substitution_num_step
            
            assert generation_step_capacity > 0, "\"reasning_step:min\" must be more than \"step_wight\" of \"Substitution (format = num)\""


            
            
            # ??????????????????(generation_step_capacity)??????????????????, ??????????????????????????????????????????
            while True:
                
                possible_assignment_format = self.get_possible_assignment_format(generation_rule["assignment_format"], generation_step_capacity, len(assignment_configs))

                
                # ??????????????????????????????, ???????????????????????????????????????????????????????????????????????? 
                if len(possible_assignment_format) == 0:
                    assert generation_step_capacity == 0, "You must include rule that has \"step_weight = 1\""
                    break
                
                
                possible_assignment_weights = [paf["probability"] for paf in possible_assignment_format]
                assignment_format = self.random_module.choices(possible_assignment_format, weights=possible_assignment_weights)[0]

                temp_symbol = shuffled_symbol_list.pop()
                # None ?????????false?????????
                commutative = bool(assignment_format.get("commutative"))


                if type(assignment_format["type"]) is list:
                    selected_assignment_type = self.random_module.choice(assignment_format["type"])
                else:
                    selected_assignment_type = assignment_format["type"]
                
                
                assignment_configs.append(
                    {
                        "variable" : temp_symbol,
                        "type" : selected_assignment_type,
                        "format" : self.instantiate_format(assignment_format["format"], selected_assignment_type, assignment_configs, commutative, generation_type="random")
                    }
                )

                # ???????????????????????????????????????????????????
                generation_step_capacity -= assignment_format["step_weight"]
                
                
                
            commutative = bool(generation_rule["operator"].get("commutative"))

            if type(generation_rule["operator"]["type"]) is list:
                selected_ope = self.random_module.choices(generation_rule["operator"]["type"], weights=generation_rule["operator"]["selection_probabilities"])[0]
            else:
                selected_ope = generation_rule["operator"]["type"]

            
            operator_config = {
                "ope"  : selected_ope,
                "format" : self.instantiate_format(generation_rule["operator"]["format"], selected_ope, assignment_configs, commutative, generation_type="random")
            }
            
            yield operator_config, assignment_configs

            

    def get_pqa_triple_from_configs(self, operator_config, assignment_configs, separate=False):
        neumeric_question = NumericQuestion(operator_config, assignment_configs, self.min_value, self.max_value, self.output_type)

        result = neumeric_question()
        if separate:
            result = [(result[0], q, a) for q, a in zip(result[1],result[2])]
        
        return result

    
            
    def __call__(self, generate_config=False):
        generator_list = []

        #??????????????????????????????????????????????????????
        for generation_rule in self.generation_rules:
            if generation_rule["type"] == "random":
                generator_list.append(self.generator_of_random_configs(generation_rule))
            elif generation_rule["type"] == "template":
                generator_list.append(self.generator_of_template_configs(generation_rule))
            else:
                error_rule_name = generation_rule["type"]
                raise Exception(f"rule \"{error_rule_name}\" is not defined")


        selection_weigths = [generation_rule["selection_probability"] for generation_rule in self.generation_rules]


        if self.max_number_of_question == "inf":
            counter = count()
        else:
            counter = range(self.max_number_of_question)
        
        
        if not generate_config: 
            for i in counter:
                temp_generator = self.random_module.choices(generator_list, weights=selection_weigths)[0]
                yield self.get_pqa_triple_from_configs(*next(temp_generator))
        else:
            for i in counter:
                yield next(self.random_module.choices(generator_list, weights=selection_weigths)[0])
        
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath",  help="Select config file", type = str)
    args = parser.parse_args()
    
    
    N = NumericDataGenarator(config_filepath=args.config_filepath)
    g = N(generate_config=True)
    for data in g:
        #passage, question, answer = data
        #print(passage)
        #print(question)
        #print(answer)
        operator_config, assignment_configs = data
        pprint(operator_config)
        pprint(assignment_configs)
        print(N.get_pqa_triple_from_configs(*data))
        
