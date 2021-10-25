import json
import itertools
import random
import string
import argparse
from copy import deepcopy
from itertools import permutations, chain


import operators
from operators import *
import reasning_operators
from reasning_operators import *


class NumericQuestion:
    def __init__(self, operator_config, assignment_config, min_value, max_value, output_type):
        self.ope = operator_config["ope"]
        assert self.ope in operators.__all__, f"operator \"{ope}\" is not defined"

        self.ope_args = operator_config["args"]

        self.default_operator_assumption = {o: globals()[o] for o in operators.__all__}
        self.reasning_operators_dict = {o: globals()[o] for o in reasning_operators.__all__}
        self.prog_env = {}
        self.prog_env.update(**self.default_operator_assumption, **self.reasning_operators_dict)

        self.assumptions = assignment_config
        self.prog_states = []
        self.final_state = None
        self.ans = None
        self.max_value = max_value
        self.min_value = min_value
        self.output_type = output_type
        self.calculated = False



        

    def exec_prog(self):
        ope_args_str = ", ".join(map(lambda s: f"{s}={s}", self.ope_args))
        

        for assumption in self.assumptions:
            # 引数となっている変数の値を保持
            args_state = {s : self.prog_env.get(s) for s in assumption["format"] if not (self.prog_env.get(s) is None)}
            self.prog_states.append(args_state)

            
            prog = "{} = round({}.assign({}), 2)".format(assumption["variable"], assumption["type"], ", ".join(assumption["format"]))

            exec(prog, {}, self.prog_env)
            

        exec(f"ans = {self.ope}.func({ope_args_str})\nans = round(ans, 2) if type(ans) is float else ans", {}, self.prog_env)
            


    def assign_value(self):

        #プログラムの実行
        self.exec_prog()


        self.ans = self.prog_env["ans"]
        self.final_state = deepcopy(self.prog_env)
        for s in chain(self.default_operator_assumption.keys(), self.reasning_operators_dict.keys()):
            self.final_state.pop(s)

        self.calculated = True



    def make_passage(self):
        passage_list = []
        for assumption, state in zip(self.assumptions, self.prog_states):
            one_substitution_passage = "{}={}".format(assumption["variable"], self.reasning_operators_dict[assumption["type"]].get_representaion(assumption["format"], state))
            passage_list.append(one_substitution_passage)

        return ",".join(passage_list)
        
        
    # 生成されたpassage, question, answerの3つ組を返す
    def make_pqa_triple(self):
        assert self.calculated, "Call assign_value meethod before this method."

        if self.output_type == "ask_last_question":
            
            passage = self.make_passage()
            assert not (self.ans is None), "Don't use No answer operation in \"ask_last_question\""
            answer = [str(self.ans)]
            question = [self.default_operator_assumption[self.ope].get_representaion(*self.ope_args) + "="]

            
            

        elif self.output_type == "ask_all_variables":
    
            passage = self.make_passage()
            self.final_state.pop("ans")
            question = [f"{k}=" for k in self.final_state.keys()]
            answer = list(map(str, self.final_state.values()))
            
            if not (self.ans is None):
                last_question = self.default_operator_assumption[self.ope].get_representaion(*self.ope_args) + "="
                question.append(last_question)
                answer.append(str(self.ans))
            
        else:
            raise NotImplementedError(f"output_type \"{self.output_type}\" is not defined!")

        return passage, question, answer


class NumericDataGenarator:
    def __init__(self, config_filepath="./config.json"):
        with open(config_filepath, mode="r") as f:
            self.config_dict = json.load(f)

        random.seed(self.config_dict["seed"])

        self.number_of_symbols = self.config_dict["number_of_symbols"]
        assert 2 <= self.number_of_symbols <= 52, "number_of_symbols is out of range"

        self.symbols = (string.ascii_uppercase + string.ascii_lowercase)[:self.number_of_symbols]
        
        self.max_number_of_question = self.config_dict["max_number_of_question"]
        self.max_value = self.config_dict["max_value"]
        self.min_value = self.config_dict["min_value"]
        self.generation_rules = self.config_dict["generation_rules"]
        
        self.reasning_operators_dict = {o : globals()[o] for o in reasning_operators.__all__}
        
        self.output_type = self.config_dict["output_type"]
        self.dtype = self.config_dict["dtype"]

        if self.dtype=="int":
            self.random_func = random.randint
        elif self.dtype=="float":
            self.random_func = lambda a, b : round(random.uniform(a, b), 2)
        else:
            raise NotImplementedError(f"Dtype \"{self.dtype}\" in config file is not defined")
        
        
    def instantiate_format(self, possible_formats, assignment_format_type, temp_assignment_configs, commutative, use_index=False):
        
        #リストの要素をタプル化
        possible_formats = list(map(lambda x: tuple(x) if type(x) is list else x, possible_formats))
        

        if commutative:
            #possible_formats += list(permutations(possible_formats))
            temp_possible_formats = []
            for pf in possible_formats:
                if not(type(pf) is tuple):
                    continue
                temp_possible_formats += list(map(tuple, permutations(pf)))

            # 同一のものを取り除く
            possible_formats = list(set(possible_formats + temp_possible_formats))
            
        
        
        
        temp_assignment_config_variables = [tac["variable"] for tac in temp_assignment_configs]
        selected_format = random.choices(possible_formats)[0]
        
        instantiated_format = []
        
        if type(selected_format) is tuple:
            format_for_assertion = tuple(map(lambda x : "var" if type(x) is int else x, selected_format))
            # 代入に用いる, 変数, 数値の形式が正しいか確認
            assert format_for_assertion in self.reasning_operators_dict[assignment_format_type].arg_formats, f"\"{assignment_format_type}\" is not support \"{format_for_assertion}\"."
            
            for elem in selected_format:
                if elem == "num":
                    instantiated_format.append(str(self.random_func(self.min_value, self.max_value)))
                elif elem == "var":
                    instantiated_format.append(random.choices(temp_assignment_config_variables)[0])
                elif (type(elem) is int) and use_index:
                    instantiated_format.append(temp_assignment_config_variables[elem])
                else:
                    raise NotImplementedError()
        
        
        else:
            format_for_assertion = "var" if type(selected_format) is int else selected_format
            assert format_for_assertion in self.reasning_operators_dict[assignment_format_type].arg_formats, f"\"{assignment_format_type}\" is not support \"{format_for_assertion}\"."

            if selected_format == "num":
                instantiated_format.append(str(self.random_func(self.min_value, self.max_value)))
            elif selected_format == "var":
                instantiated_format.append(random.choices(temp_assignment_config_variables)[0])
            elif (type(selected_format) is int) and use_index:
                instantiated_format.append(temp_assignment_config_variables[selected_format])
            else:
                raise NotImplementedError()

                
        return instantiated_format





    def instantiate_operator_args(self, operator_args_pos, assignment_configs, commutative):
        operator_args_pos = list(map(lambda x: tuple(x) if type(x) is list else x, operator_args_pos))

        if len(operator_args_pos) == 0:
            return []

        if commutative:
            temp_operator_args_pos = []
            for oap in operator_args_pos:
                if not(type(oap) is tuple):
                    continue
                temp_operator_args_pos += list(map(tuple, permutations(oap)))

            # 同一のものを取り除く
            operator_args_pos = list(set(operator_args_pos + temp_operator_args_pos))


        selected_format = random.choices(operator_args_pos)[0]
        instantiated_operator_args = []
        random_iter = iter(random.sample(range(len(assignment_configs)), len(assignment_configs)))

        if type(selected_format) is tuple:

            for elem in selected_format:
                if elem == "random":
                    instantiated_operator_args.append(assignment_configs[next(random_iter)]["variable"])
                elif type(elem) is int:
                    instantiated_operator_args.append(assignment_configs[elem]["variable"])
                else:
                    raise NotImplementedError()
        else:
            
            if selected_format == "random":
                instantiated_operator_args.append(assignment_configs[next(random_iter)]["variable"])
            elif type(selected_format) is int:
                instantiated_operator_args.append(assignment_configs[selected_format]["variable"])
            else:
                raise NotImplementedError()
        

        return instantiated_operator_args
    

    def generator_of_template(self, generation_rule):
        assert generation_rule["type"] == "template", "generation_rule's type is not match."
        #assert len(generation_rule["assignment_format"]) >= 2, "\"assignment_format\" must be longer than 2"
        
        while True:
            assignment_configs = []
            shuffled_symbol_list = random.sample(self.symbols, len(self.symbols))
        
            for assignment_format in generation_rule["assignment_format"]:
                temp_symbol = shuffled_symbol_list.pop()
                # None の時もfalseになる
                commutative = bool(assignment_format.get("commutative"))

                if type(assignment_format["type"]) is list:
                    selected_assignment_type = random.choice(assignment_format["type"])
                else:
                    selected_assignment_type = assignment_format["type"]
                    
                assignment_configs.append(
                    {
                        "variable" : temp_symbol,
                        "type" : selected_assignment_type,
                        "format" : self.instantiate_format(assignment_format["format"], selected_assignment_type, assignment_configs, commutative, use_index=True)
                    }
                )

                
            operator_args = self.instantiate_operator_args(generation_rule["operator"]["arg_pos"], assignment_configs, bool(generation_rule["operator"].get("commutative")))


            
            if type(generation_rule["operator"]["type"]) is str:
                
                operator_config = {
                    "ope"  : generation_rule["operator"]["type"],
                    "args" : operator_args
                }
                
            else:
                operator_config = {
                    "ope"  : random.choices(generation_rule["operator"]["type"], weights=generation_rule["operator"]["selection_probabilities"])[0],
                    "args" : operator_args
                }
                

            neumeric_question = NumericQuestion(operator_config, assignment_configs, self.min_value, self.max_value, self.output_type)
            neumeric_question.assign_value()
            passage, question, answer = neumeric_question.make_pqa_triple()
            
            yield passage, question, answer
        
       



    def get_possible_assignment_format(self, assignment_format_list, generation_step_capacity, number_of_available_variable):
        """
        現在, 使用可能な変数への値の割り当て方法のリストを作成する. 
        条件1 :　残り使用可能なステップ数以下である
        条件2 :　既に定義されている変数の数が, 引数で使用される変数の数よりも多い
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
                
        

        


    def generator_of_random(self, generation_rule):
        assert generation_rule["type"] == "random", "generation_rule's type is not match."
        assert all(map(lambda x: len(x["format"]) == 1, generation_rule["assignment_format"])), "assignment_format of random generation must have \"format\" that length is 1."

        
        min_generation_step = generation_rule["reasning_step"]["min"]
        max_generation_step = generation_rule["reasning_step"]["max"]
        assert min_generation_step, "\"reasning_step:min\" must be longer than 2"

        
        # 最初は必ず変数の数値代入でなければならないので, その設定は(生成確率0でも良いので)configに含める. 
        substitution_num_step = next(filter(lambda x: x["type"] == "Substitution" and x["format"] == ["num"], generation_rule["assignment_format"]))["step_weight"]


        # yieldのループ
        while True:
            shuffled_symbol_list = random.sample(self.symbols, len(self.symbols))


            possible_format_list = generation_rule["assignment_format"]
            possible_format_selection_probability = [pf["probability"] for pf in possible_format_list]
            generation_step_capacity = random.randint(min_generation_step, max_generation_step)

            # 最初は必ず変数の数値代入でなければならない
            assignment_configs = [
                {
                    "variable" : shuffled_symbol_list.pop(),
                    "type" : "Substitution",
                    "format" : [str(self.random_func(self.min_value, self.max_value))]
                }
            ]         
            generation_step_capacity -= substitution_num_step
            
            assert generation_step_capacity > 0, "\"reasning_step:min\" must be more than \"step_wight\" of \"Substitution (format = num)\""


            
            
            # ステップ数が(generation_step_capacity)が尽きるまで, 変数の割り当て方法を決定する
            while True:
                
                possible_assignment_format = self.get_possible_assignment_format(generation_rule["assignment_format"], generation_step_capacity, len(assignment_configs))

                
                # ステップ数を使い切り, これ以上割り当てを行えなくなったらループを抜ける 
                if len(possible_assignment_format) == 0:
                    assert generation_step_capacity == 0, "You must include rule that has \"step_weight = 1\""
                    break
                
                
                possible_assignment_weights = [paf["probability"] for paf in possible_assignment_format]
                assignment_format = random.choices(possible_assignment_format, weights=possible_assignment_weights)[0]

                temp_symbol = shuffled_symbol_list.pop()
                # None の時もfalseになる
                commutative = bool(assignment_format.get("commutative"))


                if type(assignment_format["type"]) is list:
                    selected_assignment_type = random.choice(assignment_format["type"])
                else:
                    selected_assignment_type = assignment_format["type"]
                
                
                assignment_configs.append(
                    {
                        "variable" : temp_symbol,
                        "type" : selected_assignment_type,
                        "format" : self.instantiate_format(assignment_format["format"], selected_assignment_type, assignment_configs, commutative, use_index=False)
                    }
                )

                # 使用可能な残りのステップ数を減らす
                generation_step_capacity -= assignment_format["step_weight"]
                
                

            
            operator_args = self.instantiate_operator_args(generation_rule["operator"]["arg_pos"], assignment_configs, bool(generation_rule["operator"].get("commutative")))

            
            if type(generation_rule["operator"]["type"]) is str:
                
                operator_config = {
                    "ope"  : generation_rule["operator"]["type"],
                    "args" : operator_args
                }
                
            else:
                operator_config = {
                    "ope"  : random.choices(generation_rule["operator"]["type"], weights=generation_rule["operator"]["selection_probabilities"])[0],
                    "args" : operator_args
                }


            
            neumeric_question = NumericQuestion(operator_config, assignment_configs, self.min_value, self.max_value, self.output_type)
            neumeric_question.assign_value()
            passage, question, answer = neumeric_question.make_pqa_triple()
            
            yield passage, question, answer    

            



    
            
    def generate_data(self):
        generator_list = []

        #各ルールに基づいたジェネレータの作成
        for generation_rule in self.generation_rules:
            if generation_rule["type"] == "random":
                generator_list.append(self.generator_of_random(generation_rule))
            elif generation_rule["type"] == "template":
                generator_list.append(self.generator_of_template(generation_rule))
            else:
                error_rule_name = generation_rule["type"]
                raise Exception(f"rule \"{error_rule_name}\" is not defined")


        selection_weigths = [generation_rule["selection_probability"] for generation_rule in self.generation_rules]

            
        for i in range(self.max_number_of_question):
            temp_generator = random.choices(generator_list, weights=selection_weigths)[0]
            yield next(temp_generator)


                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath",  help="Select config file", type = str)
    args = parser.parse_args()
    
    N = NumericDataGenarator(config_filepath=args.config_filepath)
    g = N.generate_data()
    for triple in g:
        print(triple)