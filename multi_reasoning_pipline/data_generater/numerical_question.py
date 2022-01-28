from copy import deepcopy
from itertools import chain

import operators
from operators import *

class NumericQuestion:
    def __init__(self, operator_config, assignment_config, output_type):
        
        self.ope = operator_config["ope"]
        assert self.ope in operators.__all__, f"operator \"{ope}\" is not defined"

        self.ope_args = operator_config["format"]

        self.operators_dict = {o: globals()[o] for o in operators.__all__}
        self.prog_env = {}
        self.prog_env.update(**self.operators_dict)

        self.assumptions = assignment_config
        self.prog_states = []
        self.substitution_operator_dict = {}
        self.final_state = None
        self.ans = None
        self.output_type = output_type
        self.calculated = False



        

    def exec_program(self):

        for i, assumption in enumerate(self.assumptions):
            
            # 引数となっている変数の値を保持
            args_state = {s : self.prog_env.get(s) for s in assumption["format"] if self.prog_env.get(s) is not None}
            
            self.prog_states.append(args_state)
           
            variable = assumption["variable"]
            
            self.prog_env["state"] = args_state
            self.prog_env["arg_list"] = assumption["format"]
            
            prog = f"""ope{i} = {assumption["type"]}()\n{variable} = ope{i}(arg_list, state)\n{variable} = round({variable}, 2) if type({variable}) is float else {variable}"""
            
            
            exec(prog, {}, self.prog_env)
            
            self.substitution_operator_dict[f"ope{i}"] = self.prog_env[f"ope{i}"]

        
        args_state = {s : self.prog_env.get(s) for s in self.ope_args if not (self.prog_env.get(s) is None)}
        
        self.prog_states.append(args_state)
        self.prog_env["state"] = args_state
        self.prog_env["arg_list"] = self.ope_args
        
        prog = f"ope_last = {self.ope}();ans = ope_last(arg_list, state);ans = round(ans, 2) if type(ans) is float else ans"
        
        exec(prog, {}, self.prog_env)
            
        self.substitution_operator_dict["ope_last"] = self.prog_env["ope_last"]
        


    def assign_value(self):

        #プログラムの実行
        self.exec_program()
        
        self.ans = self.prog_env["ans"]
        self.final_state = deepcopy(self.prog_env)
        
        # プログラムの状態から余分なクラス, 変数定義の削除
        for s in chain(self.operators_dict.keys(), self.substitution_operator_dict.keys()):
            self.final_state.pop(s)
        self.final_state.pop("state")
        self.final_state.pop("arg_list")

        self.calculated = True



    def make_passage(self):
        passage_list = []

        
        for i, (assumption, state) in enumerate(zip(self.assumptions, self.prog_states[:-1])):
            
            one_substitution_passage = "{} = {}".format(
                assumption["variable"],
                self.substitution_operator_dict[f"ope{i}"].get_representation(
                    assumption["format"],
                    state
                )
            )
            
            passage_list.append(one_substitution_passage)

        return ", ".join(passage_list)
        
        
    # 生成されたpassage, question, answerの3つ組を返す
    def make_pqa_triple(self):
        assert self.calculated, "Call assign_value meethod before this method."

        if self.output_type == "ask_last_question":
            
            passage = self.make_passage()
            assert not (self.ans is None), "Don't use No answer operation in \"ask_last_question\""
            answer = [str(self.ans)]
            question = [self.substitution_operator_dict["ope_last"].get_representation(self.ope_args, self.prog_states[-1]) + " ="]
            
            
            

        elif self.output_type == "ask_all_variables":
    
            passage = self.make_passage()
            self.final_state.pop("ans")
            question = [f"{k} =" for k in self.final_state.keys()]
            answer = list(map(str, self.final_state.values()))

            if not (self.ans is None):
                last_question = self.substitution_operator_dict["ope_last"].get_representation(self.ope_args, self.prog_states[-1]) + " ="
                question.append(last_question)
                answer.append(str(self.ans))

        else:
            raise NotImplementedError(f"output_type \"{self.output_type}\" is not defined!")

        return passage, question, answer



    def __call__(self):
        self.assign_value()
        return self.make_pqa_triple()
    
