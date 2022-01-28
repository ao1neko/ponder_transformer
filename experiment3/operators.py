from functools import reduce

from basic_operator import BasicOperator
from operator_tools import *

"""
 定義したクラスを必ず追加してください. (operaterとして用いない場合は不要です. )
"""
__all__ = ["Substitution", "Check", "Add", "Sub", "WhoMax", "PseudoAdd", "NoQuestion", "Max", "Min", "AND", "OR", "NAND", "NOR", "XOR", "Implication"]





class Substitution(BasicOperator):
    arg_formats = ["num", "var"]

    def __call__(self, arg_list, state):
        assert len(arg_list) == 1, "lehgth of arg_list must be 1 when you use \"Substitution\"."
        return get_values(arg_list, state)[0]
    

    def get_representation(self, arg_list, state):
        return arg_list[0]

    
class Check(Substitution):
    arg_formats = ["var"]

    
class Add(BasicOperator):
    """
    f"*num:{n}", f"*var:{n}", f"*num_var:{n}"は(n >= 1)個以上の任意の個数の数値, 変数, 数値と変数の両方を受け取れることを示すワイルドカード的なものです.
    
    例:

    arg_formats = ["*num:1"]: 1個以上の任意の個数の数値
                = ["num", ("num", "num"), ("num", "num", "num"),...]
    arg_formats = ["*num_var:2"] : 2個以上の任意の個数の数値 or 変数
                = [("num", "var"), ...]
    (変数と数値が引数に混ざっていてもOKです.)

    arg_formats = ["*num:1", "*var:1"] : 1個以上の任意の個数の数値 or 1個以上の任意の個数の変数
                = ["num", ("num", "num"), ,... , "var", ("var", "var")]
    (変数と数値が混ざっているものは受け取れないことを意味します. あまり使い道はないかもしれませんが,  一応そういう仕様にします. )
    

    現状, このワイルドカードは単体でしか用いることができないです. 
    こういった使い方はできないです. 
    ダメな例 : 
    arg_formats = [("*var", "num")] : 1個以上の任意の個数の変数と, 1つの数値
    
    もちろん, 今まで通り個別に形式を指定できます. 
    例：
    arg_formats = [("num", "num"), ("var", "num"), ("num", "var"), ("var", "var")]
 
    """
    arg_formats = ["*num_var:2"]
    
    def __call__(self, arg_list, state):
        #assert len(arg_list) == 2
        return sum(get_values(arg_list, state))
    

    def get_representation(self, arg_list, state):
        return " + ".join(map(num2str, arg_list))



class Sub(BasicOperator):
    arg_formats = ["*num_var:2"]
    
    def __call__(self, arg_list, state):
        #assert len(arg_list) == 2
        return func_nest(lambda a, b: a - b, get_values(arg_list, state))
    

    def get_representation(self, arg_list, state):
        return " - ".join(map(num2str, arg_list))





    
class Max(BasicOperator):
    arg_formats = ["*num_var:1"]
    
    def __call__(self, arg_list, state):
        return max(get_values(arg_list, state))
    

    def get_representation(self, arg_list, state):
        return " ^ ".join(map(num2str, arg_list))


    

class Min(BasicOperator):
    arg_formats = ["*num_var:1"]
    
    def __call__(self, arg_list, state):
        return max(get_values(arg_list, state))
    

    def get_representation(self, arg_list, state):
        return " * ".join(map(num2str, arg_list))




    
    
class WhoMax(BasicOperator):
    arg_formats = ["*var:1"]
    
    def __call__(self, arg_list, state):
        return sorted(zip(arg_list, get_values(arg_list, state)), key=lambda x: x[1], reverse=True)[0][0]
    

    def get_representation(self, arg_list, state):
        return "who_max({})".format(",".join(map(num2str, arg_list)))
    

    
class PseudoAdd(Add):
    
    def get_representation(self, arg_list, state):
        return " + ".join(map(num2str, get_values(arg_list, state)))



class NoQuestion(BasicOperator):
    arg_formats = []

    def __call__(self, arg_list, state):
        assert len(arg_list) == 0 and len(state) == 0, "This operator has no arguments."
        return None
    

    def get_representation(self, arg_list, state):
        return None



    
class OR(BasicOperator):
    arg_formats = ["*num_var:2"]
    
    def __call__(self, arg_list, state):
        values = get_values(arg_list, state)
        assert all(map(lambda x: x == 0 or x == 1, values)), "Not bool number..."
        return int(reduce(lambda a, b: a or b, values))
    

    def get_representation(self, arg_list, state):
        return " | ".join(map(num2str, arg_list))


class AND(BasicOperator):
    arg_formats = ["*num_var:2"]
    
    def __call__(self, arg_list, state):
        values = get_values(arg_list, state)
        assert all(map(lambda x: x == 0 or x == 1, values)), "Not bool number..."
        return int(reduce(lambda a, b: a and b, values))
    

    def get_representation(self, arg_list, state):
        return " & ".join(map(num2str, arg_list))


class XOR(BasicOperator):
    arg_formats = ["*num_var:2"]
    
    def __call__(self, arg_list, state):
        values = get_values(arg_list, state)
        assert all(map(lambda x: x == 0 or x == 1, values)), "Not bool number..."
        return int(reduce(lambda a, b: a ^ b, values))
    

    def get_representation(self, arg_list, state):
        return " ^ ".join(map(num2str, arg_list))




class NOR(OR):
    arg_formats = ["*num_var:2"]
    
    def __call__(self, arg_list, state):
        return int(not super().__call__(arg_list, state))    

    def get_representation(self, arg_list, state):
        return " + ".join(map(num2str, arg_list))



class NAND(AND):
    arg_formats = ["*num_var:2"]
    
    def __call__(self, arg_list, state):
        return int(not super().__call__(arg_list, state))

    def get_representation(self, arg_list, state):
        return " * ".join(map(num2str, arg_list))


class Implication(BasicOperator):
    arg_formats = ["*num_var:2"]

    def __call__(self, arg_list, state):
        values = get_values(arg_list, state)
        assert all(map(lambda x: x == 0 or x == 1, values)), "Not bool number..."
        return int(reduce(lambda a, b: (not a) or b, values))
    
    
    def get_representation(self, arg_list, state):
        return " > ".join(map(num2str, arg_list))











    
