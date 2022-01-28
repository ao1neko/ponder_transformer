def numstr2num(s):
    num = float(s)

    try:
        num = int(s)
    except ValueError:
        pass
    
    return num


def num2str(num_or_valiable: str) -> str:
    try:
        num = numstr2num(num_or_valiable)
        return str(num)
    except ValueError:
        return num_or_valiable

    """
    if num < 0:
        return f"( - {str(abs(num))} )" 
    else:
        return str(num)
    """
    


def get_values(arg_list, state):
    values = []
    for arg in arg_list:
        num = state.get(arg)
        if num is None:
            try:
                num = int(arg)
            except ValueError:
                num = float(arg)
        values.append(num)

    return values

    

def func_nest(f, iterable):
    input_iter = iter(iterable)
    temp_value = next(input_iter)
    for v in input_iter:
        temp_value = f(temp_value, v)
    return temp_value
