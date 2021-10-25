import random
from substitution_func_wrapper import SubstitutionFuncWrapper


"""
 定義したクラスを必ず追加してください. (operationとして用いない場合は不要です. )
 """
__all__ = ["Substitution", "AddSubstitution", "SubSubstitution", "PseudoAddSubstitution", "HalfLeftPseudoAddSubstitution", "HalfRightPseudoAddSubstitution", "HalfLeftPseudoSubSubstitution", "HalfRightPseudoSubSubstitution"]

class Substitution(SubstitutionFuncWrapper):
    """
    推論時の操作の内容を表すクラスです. 
    必ずSubstitutionFuncWrapperを継承してください.
    arg_formats, assignメソッドとget_representaionメソッドを必ず定義する必要があります。
    """
    
    """
    assignメソッドが取ることができる引数の形式を設定します. 
    リストの各要素が取ることのできる引数の形式で, 
    "num" : 数値
    "var" : 変数
    を表します. 
    この内容はランダムにデータを構成する際などに, 矛盾なくデータを構成するために用いられます. 
    """
    arg_formats = ["num", "var"]


    def assign(arg):
        """
        変数への代入を行う際の操作の内容を定義します. 
        Substitutionクラスでは, 数値or変数を受け取って単純に代入を行うので, 入力をそのままreturnしています. 
        """
        return arg

    def get_representaion(arg_list, state):
        """
        代入操作のデータセット内での表現方法(表示形式)を出力してください. 
        入力は, 関数に入力された変数の名前 or 代入された数値(str)です. 
        例えば, A = Substitution.assign(B), A = Substitution.assign(1)と処理されていた場合, 
        出力は, A=B, A=1のようになります. 
        """
        assert len(arg_list) == 1, "lehgth of arg_list must be 1."
        return arg_list[0]

    

class AddSubstitution(SubstitutionFuncWrapper):
    """
    複数の引数をとる場合は, arg_formatsの各要素はタプルにしてください. 
    """
    arg_formats = [("num", "num"), ("var", "num"), ("num", "var"), ("var", "var")]
    
    def assign(arg1, arg2):
        return arg1 + arg2

    def get_representaion(arg_list, state):
        return "+".join([('('+arg+')' if arg[0]=='-' else arg) for arg in arg_list])


class SubSubstitution(SubstitutionFuncWrapper):
    """
    複数の引数をとる場合は, arg_formatsの各要素はタプルにしてください. 
    """
    arg_formats = [("num", "num"), ("var", "num"), ("num", "var"), ("var", "var")]
    
    def assign(arg1, arg2):
        return arg1 - arg2

    def get_representaion(arg_list, state):
        return "-".join([('('+arg+')' if arg[0]=='-' else arg) for arg in arg_list])
    

class PseudoAddSubstitution(SubstitutionFuncWrapper):
    """
    変数で代入されたものを「全て」その変数の持つ数値に置換した表現で出力します. 
    (このクラスの, いい名前を募集してます!)
    """
    arg_formats = [("num", "num"), ("var", "num"), ("num", "var"), ("var", "var")]
    
    def assign(arg1, arg2):
        return arg1 + arg2

    def get_representaion(arg_list, state):
        return "+".join([str(state.get(arg)) if not(state.get(arg) is None) else arg for arg in arg_list])



    
class HalfLeftPseudoAddSubstitution(SubstitutionFuncWrapper):
    arg_formats = [("var", "var")]
    
    def assign(arg1, arg2):
        return arg1 + arg2

    def get_representaion(arg_list, state):
        assert len(arg_list)==2, "引数の長さが違います"
        arg_list[0] = str(state.get(arg_list[0]))
        return "+".join([('('+arg+')' if arg[0]=='-' else arg) for arg in arg_list])


class HalfRightPseudoAddSubstitution(SubstitutionFuncWrapper):
    arg_formats = [("var", "var")]
    
    def assign(arg1, arg2):
        return arg1 + arg2

    def get_representaion(arg_list, state):
        assert len(arg_list)==2, "引数の長さが違います"
        arg_list[1] = str(state.get(arg_list[1]))
        return "+".join([('('+arg+')' if arg[0]=='-' else arg) for arg in arg_list])



    
class HalfLeftPseudoSubSubstitution(SubstitutionFuncWrapper):
    arg_formats = [("var", "var")]
    
    def assign(arg1, arg2):
        return arg1 - arg2

    def get_representaion(arg_list, state):
        assert len(arg_list)==2, "引数の長さが違います"
        arg_list[0] = str(state.get(arg_list[0]))
        return "-".join([('('+arg+')' if arg[0]=='-' else arg) for arg in arg_list])    


class HalfRightPseudoSubSubstitution(SubstitutionFuncWrapper):
    arg_formats = [("var", "var")]
    
    def assign(arg1, arg2):
        return arg1 - arg2

    def get_representaion(arg_list, state):
        assert len(arg_list)==2, "引数の長さが違います"
        arg_list[1] = str(state.get(arg_list[1]))
        return "-".join([('('+arg+')' if arg[0]=='-' else arg) for arg in arg_list])    



    