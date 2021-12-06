from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasicOperator(ABC):
    arg_format = None
    

    @abstractmethod
    def __call__(self, arg_list: List[str], state: Dict[str, Any]):
        raise NotImplementedError()

    @abstractmethod
    def get_representation(self, arg_list: List[str], state: Dict[str, Any]):
        raise NotImplementedError()

    def get_values(self, arg_list, state):
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

            

        
