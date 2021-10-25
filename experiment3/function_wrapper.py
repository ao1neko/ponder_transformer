from abc import ABCMeta, abstractmethod

class FunctionWrapper(ABCMeta):

    @classmethod
    @abstractmethod
    def func(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_representaion(cls, *symbols):
        raise NotImplementedError()



if __name__ == "__main__":
    print(str(FunctionWrapper))


    






