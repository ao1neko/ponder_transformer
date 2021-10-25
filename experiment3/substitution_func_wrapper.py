from abc import ABCMeta, abstractmethod


class SubstitutionFuncWrapper(ABCMeta):
    arg_format = None
    step_weight = None

    @classmethod
    @abstractmethod
    def assign(args):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_representaion(*args):
        raise NotImplementedError()
