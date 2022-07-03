import argparse
from cmath import e, log
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
from torch.utils.data.dataset import Subset
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List, Dict, Set
import string
import copy
import json

class Equation():
    def __init__(self, left_side=[], right_side=[], left_char_set=set(), right_char_set=set()):
        self.left_side = left_side
        self.right_side = right_side
        self.left_char_set = left_char_set
        self.right_char_set = right_char_set

    def calc_char_set(self):
        self.left_char_set = {x for x in self.left_side if type(x) is str}
        self.right_char_set = {x for x in self.right_side if type(x) is str}

    def calc_side_int_sum(self, calc_side="left"):
        if calc_side == "left":
            return [x for x in self.left_side if type(x) is int].sum()
        elif calc_side == "right":
            return [x for x in self.right_side if type(x) is int].sum()
        
        

class DataInstance():
    def __init__(self, inference_step: int, equation_num: int, char_set: Set = set(string.ascii_uppercase)):
        self.equations = []
        self.forward_inference_equations = []
        self.forward_backtrack_inference_equations = []
        self.backward_inference_equations = []
        self.question = None
        self.answer = None
        self.char_set = set()
        self.useable_char_set = char_set


    def to_str(self):
        equations_str = " , ".join([" + ".join(x[0]) + " " + " + ".join(x[1]) for x in self.equations])
        forward_inference_equations_str = " , ".join([" + ".join(x[0]) + " " + " + ".join(x[1]) for x in self.forward_inference_equations])
        forward_backtrack_inference_equations_str = " , ".join([" + ".join(x[0]) + " " + " + ".join(x[1]) for x in self.forward_backtrack_inference_equations])
        backward_inference_equations_str = " , ".join([" + ".join(x[0]) + " " + " + ".join(x[1]) for x in self.backward_inference_equations])
        question_str = self.question + " ? "
        answer_str = str(self.answer)
        return (equations_str,question_str,answer_str,forward_inference_equations_str,forward_backtrack_inference_equations_str,backward_inference_equations_str)
        
        

    def to_json(self):
        json_data = {
            "equations" : self.equations,
            "question" : self.question,
            "answer" : self.answer,
            "inference_equations" : {
                "forward_inference_equations" : self.forward_inference_equations,
                "forward_backtrack_inference_equations" : self.forward_backtrack_inference_equations,
                "backward_inference_equations" : self.backward_inference_equations,
            },
            "inform" : {
                "char_set" : self.char_set,
                "useable_char_set" : self.useable_char_set,
            }
        }
        return json_data
    
    def from_json(self,json_data:str):
        json_dict = json.loads(json_data)
        self.equations = json_dict["equations"]
        self.forward_inference_equations = json_dict["inference_equations"]["forward_inference_equations"]
        self.forward_backtrack_inference_equations = json_dict["inference_equations"]["forward_backtrack_inference_equations"]
        self.backward_inference_equations = json_dict["inference_equations"]["backward_inference_equations"]
        self.question =json_dict["question"]
        self.answer = json_dict["answer"] 
        self.char_set = json_dict["inform"]["char_set"]
        self.useable_char_set = json_dict["inform"]["useable_char_set"]

    def make_instance(self, inference_step: int, equation_num: int):
        self._make_minimum_instance(inference_step)
        for _ in range(equation_num - inference_step):
            self._make_relate_equation()

        self._shuffle()
        self._make_forward_inference_equations()
        self._make_forward_backtrack_inference_equations()
        self._make_backward_inference_equations()

    def _shuffle(self):
        random.shuffle(self.equations)

    def _make_minimum_instance(self,
                               inference_step: int,):
        useable_char_set = self.useable_char_set.copy()
        self.question = random.choice(useable_char_set)
        self.answer = 0
        use_char_set = set(self.question)
        use_char_set_num = len(use_char_set)

        useable_char_set = useable_char_set - use_char_set

        for index in range(inference_step):
            remain_equation_num = inference_step - index

            if remain_equation_num == use_char_set_num:
                equation_right_int_num = 1
                equation_right_char_num = 0
            elif remain_equation_num-1 < use_char_set_num+1:
                equation_right_int_num = 1
                equation_right_char_num = 1
            else:
                if random.random() < 0.1:
                    equation_right_int_num = 0
                    equation_right_char_num = 2
                else:
                    equation_right_int_num = 1
                    equation_right_char_num = 1

            left_char = set(use_char_set.pop())
            equation = self._make_equation(
                equation_left_char_set=set(left_char),
                equation_right_char_set=useable_char_set,
                equation_left_int_num=0,
                equation_left_char_num=1,
                equation_right_int_num=equation_right_int_num,
                equation_right_char_num=equation_right_char_num,
            )
            for arg in equation.right_side:
                if type(arg) is int:
                    self.answer += arg

            self.equations.append(equation)
            use_char_set.add(equation.right_char_set)
            useable_char_set = useable_char_set - equation.right_char_set

            self.char_set = self.char_set | equation.left_char_set | equation.right_char_set

    def _make_relate_equation(self,):
        useable_char_set = self.useable_char_set.copy() - self.char_set

        if len(self.char_set) >= 2 and random.random() < 0.1:
            equation_right_int_num = 0
            equation_right_char_num = 2
        else:
            equation_right_int_num = 1
            equation_right_char_num = 1

        related_equation = self._make_equation(
            equation_left_char_set=useable_char_set,
            equation_right_char_set=self.char_set,
            equation_left_int_num=0,
            equation_left_char_num=1,
            equation_right_int_num=equation_right_int_num,
            equation_right_char_num=equation_right_char_num,
        )
        self.equations.append(related_equation)

    def _make_equation(self,
                       equation_left_char_set: Set = set(
                           string.ascii_uppercase),
                       equation_right_char_set: Set = set(
                           string.ascii_uppercase),
                       equation_left_int_set: Set = set(
                           range(100)
                       ),
                       equation_right_int_set: Set = set(
                           range(100)
                       ),
                       equation_left_int_num: int = 0,
                       equation_left_char_num: int = 1,
                       equation_right_int_num: int = 1,
                       equation_right_char_num: int = 0):
        equation = Equation()
        equation_left_chars = random.choices(
            equation_left_char_set, equation_left_char_num)
        equation_right_chars = random.choices(
            equation_right_char_set, equation_right_char_num)

        equation.left_side = equation_left_chars + \
            random.choices(equation_left_int_set, equation_left_int_num)
        equation.right_side = equation_right_chars + \
            random.choices(equation_right_int_set, equation_right_int_num)
        equation.calc_char_set()

        return equation

    def _search_relate_equation(self, search_char: str, search_side: str = "left", equations="equations"):
        if equations == "equations":
            equations = self.equations
        elif equations == "forward_inference_equations":
            equations = self.forward_inference_equations
        elif equations == "forward_backtrack_inference_equations":
            equations = self.forward_backtrack_inference_equations
        elif equations == "backward_inference_equations":
            equations = self.backward_inference_equations

        if search_side == "left":
            for equation in equations:
                if search_char in self.left_side:
                    return equation
        elif search_side == "right":
            for equation in equations:
                if search_char in self.right_side:
                    return equation
        return None

    def _make_solved_char_dict(self):
        solved_char_dict = dict()
        for equation in self.equations:
            if len(equation.right_side) == 1:
                key = equation.left_side[0]
                value = equation.right_side[0]
                solved_char_dict[key] = value
        return solved_char_dict

    def _make_forward_inference_equations(self):
        solved_char_dict = self._make_solved_char_dict()
        question_stack = []
        not_search_question_stack = []
        not_search_question_stack.append(self.question)

        while len(not_search_question_stack) != 0:
            search_char = not_search_question_stack.pop()
            equation = self.search_relate_equation(
                search_char=self.question, search_side="left")
            search_char_list = [
                x for x in equation.right_side if type(x) is str]

            question_stack.append(search_char)
            not_search_question_stack.extend(search_char_list)

        while len(question_stack) != 0:
            for question in question_stack.copy():
                if question in solved_char_dict.keys():
                    question_stack.remove(question)
                else:
                    equation = self._search_relate_equation(
                        search_char=question, search_side="left")
                    forward_flag = True
                    for search_char in equation.right_char_set:
                        if search_char not in solved_char_dict.keys():
                            forward_flag = False
                            break
                    if forward_flag == True:
                        self.forward_inference_equations.append(equation)
                        while question not in solved_char_dict.keys():
                            equation = self.forward_inference_equations[-1].copy(
                            )
                            if len(equation.right_char_set) == 0:
                                if len(equation.right_side) == 1:
                                    key = equation.left_side[0]
                                    value = equation.right_side[0]
                                    solved_char_dict[key] = value
                                else:
                                    equation.right_side = [
                                        equation.calc_side_int_sum(calc_side="right")]
                                    self.forward_inference_equations.append(
                                        equation)
                            else:
                                search_char = [
                                    x for x in equation.right_side if type(x) is str][0]
                                replace_index = equation.right_side.index(
                                    search_char)
                                equation.right_side[replace_index] = solved_char_dict[search_char]
                                self.forward_inference_equations.append(
                                    equation)

    def _make_forward_backtrack_inference_equations(self):
        solved_char_dict = self._make_solved_char_dict()

        while self.question not in solved_char_dict.keys():
            for equation in self.equations:
                forward_flag = True
                for search_char in equation.right_char_set:
                    if search_char not in solved_char_dict.keys():
                        forward_flag = False
                        break
                if forward_flag == True:
                    self.forward_backtrack_inference_equations.append(equation)
                    search_char = equation.left_side[0]
                    while search_char not in solved_char_dict.keys():
                        equation = self.forward_inference_equations[-1].copy()
                        if len(equation.right_char_set) == 0:
                            if len(equation.right_side) == 1:
                                key = equation.left_side[0]
                                value = equation.right_side[0]
                                solved_char_dict[key] = value
                            else:
                                equation.right_side = [
                                    equation.calc_side_int_sum(calc_side="right")]
                                self.forward_backtrack_inference_equations.append(
                                    equation)
                        else:
                            search_char = [
                                x for x in equation.right_side if type(x) is str][0]
                            replace_index = equation.right_side.index(
                                search_char)
                            equation.right_side[replace_index] = solved_char_dict[search_char]
                            self.forward_backtrack_inference_equations.append(
                                equation)

    def _make_backward_inference_equations(self):
        solved_char_dict = self._make_solved_char_dict()
        goal_equation = self._search_relate_equation(
            search_char=self.question, search_side="left")
        self.backward_inference_equations.append(goal_equation)

        while self.question not in solved_char_dict.keys():
            backward_equation = self.backward_inference_equations[-1]
            if len(backward_equation.right_char_set) == 0:
                if len(backward_equation.right_side) == 1:
                    search_char = backward_equation.left_side[0]
                    equation = self.backward_inference_equations.search_relate_equation(
                        search_char=search_char, search_side="right", equations="backward_inference_equations")
                    replace_index = equation.right_side.index(search_char)
                    equation.right_side[replace_index] = solved_char_dict[search_char]
                    self.backward_inference_equations.append(equation)
                else:
                    equation = backward_equation.copy()
                    equation.right_side = [
                        equation.calc_side_int_sum(calc_side="right")]
                    key = equation.left_side[0]
                    value = equation.right_side[0]
                    solved_char_dict[key] = value
                    self.backward_inference_equations.append(equation)
            else:
                search_char = [
                    x for x in backward_equation.right_side if type(x) is str][0]
                if search_char in solved_char_dict.keys():
                    equation = backward_equation.copy()
                    replace_index = equation.right_side.index(search_char)
                    equation.right_side[replace_index] = solved_char_dict[search_char]
                    self.backward_inference_equations.append(equation)
                else:
                    equation = self._search_relate_equation(
                        search_char=search_char, search_side="left")
                    self.backward_inference_equations.append(equation)
