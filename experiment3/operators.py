from function_wrapper import FunctionWrapper
import random

# 定義したクラスを追加してください. (operationとして用いない場合は不要です. )
__all__ = ["Add", "Sub", "Mult", "Max", "Min", "WhoMax", "WhoMin", "Condition", "Check", "NoQuestion"]


# クラス名はconfigに設定する名前と同一にしてください.
class Max(FunctionWrapper):
    """
    最終的な演算の内容を表すクラスです. 
    必ずFunctionWrapperを継承してください. 
    """
    
    def func(**kwargs):
        """
        関数の計算内容を記述する部分です. 
        kwargsには, 変数の名前とその値がペアになった辞書が入力されます. 
        出力は, 数字 or stringにするのが無難です. 厳密にはここで出力された内容に, str関数を適応したものが, データセットのanswerとなります. 
        """
        return max(kwargs.values())

    def get_representaion(*symbols):
        """
        関数のデータセット内での表現方法(表示形式)を出力してください. 
        入力symbolsは, 関数に入力された変数の名前です. 
        例えば, pythonの内部的な計算ではAdd.func(A, B)などと計算が行われますが, データセットではA + Bと表現したいときもあると思うので, その設定のために用いられます. 
        """
        
        return "Max({})".format(", ".join(symbols))



class Min(FunctionWrapper):
    def func(**kwargs):
        return min(kwargs.values())

    def get_representaion(*symbols):
        return "Min({})".format(", ".join(symbols))


class Add(FunctionWrapper):
    def func(**kwargs):
        return sum(kwargs.values())

    def get_representaion(*symbols):
        return "+".join(symbols)


class Sub(FunctionWrapper):
    def func(**kwargs):
        return Sub.func_nest(lambda a, b: a - b, kwargs.values())
    
    def get_representaion(*symbols):
        return "-".join(symbols)

    def func_nest(f, iterable):
        input_iter = iter(iterable)
        temp_value = next(input_iter)
        for v in input_iter:
            temp_value = f(temp_value, v)
        return temp_value


class Mult(FunctionWrapper):
    def func(**kwargs):
        return Mult.func_nest(lambda a, b: a * b, kwargs.values())
    
    def get_representaion(*symbols):
        return "-".join(symbols)

    def func_nest(f, iterable):
        input_iter = iter(iterable)
        temp_value = next(input_iter)
        for v in input_iter:
            temp_value = f(temp_value, v)
        return temp_value

    
    
class WhoMax(FunctionWrapper):
    def func(**kwargs):
        return sorted(kwargs.items(), key = lambda x: x[1])[-1][0]
        

    def get_representaion(*symbols):
        return "WhoMax({})".format(", ".join(symbols))

class WhoMin(FunctionWrapper):
    def func(**kwargs):
        return sorted(kwargs.items(), key = lambda x: x[1])[0][0]
        

    def get_representaion(*symbols):
        return "WhoMax({})".format(", ".join(symbols))


    
class Condition(FunctionWrapper):

    threshold = None
    
    def func(**kwargs):
        Condition.threshold = random.randrange(5)
        return "{" + ", ".join(s for s, v in kwargs.items() if v >= Condition.threshold) + "}"
        

    def get_representaion(*symbols):
        return "{X | X >= " + str(Condition.threshold) + "}"



class Check(FunctionWrapper):
    def func(**kwargs):
        assert len(kwargs)==1, "lejgth of kwargs at Check must be 1."
        return next(iter(kwargs.values()))

    def get_representaion(*symbols):
        return symbols[0]

class NoQuestion(FunctionWrapper):
    arg_formats = []
    
    def func():
        return None

    def get_representaion(arg_list, state):
        return None
