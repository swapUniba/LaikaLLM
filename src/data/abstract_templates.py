from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from requests.structures import CaseInsensitiveDict


class PromptTarget:

    def __init__(self, input_prompt: str, target_text: str):
        self.input_prompt = input_prompt
        self.target_text = target_text

    # iter just so that this class can be unpacked,
    # e.g. input_prompt, target_text = PromptTarget(...)
    def __iter__(self):
        return iter((self.input_prompt, self.target_text))

    def __str__(self):
        string = " Input ".center(50, "#") + "\n"
        string += self.input_prompt + "\n"
        string += " Target ".center(50, "#") + "\n"
        string += self.target_text

        return string


class Task(ABC):

    # keys are integers, values are PromptTarget objects
    templates_dict = {}

    # name obj class mapping, used for when task must be initialized from strings
    str_alias_cls: dict[str, type[Task]] = CaseInsensitiveDict()

    # class attribute since if the model is in training mode, all tasks should be in training mode
    training: bool = False

    all_unique_items: np.ndarray = np.array([])

    def __init__(self, all_unique_items: np.ndarray[str]):
        Task.all_unique_items = all_unique_items

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):
        cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    def all_templates(self, return_id: bool = False):
        return list(self.templates_dict.keys()) if return_id else list(self.templates_dict.values())

    @abstractmethod
    def valid_templates(self, return_id: bool = False):
        raise NotImplementedError

    def force_template(self, force_template_id: int):

        # 'self.__class__' so that even if we call multiple times this method on an instantiated task,
        # we always have a pointer to original class templates, otherwise they are deleted if we use simply 'self'
        # instead of self.__class__

        if force_template_id not in set(self.__class__.templates_dict.keys()):
            raise KeyError(f"Prompt template id {force_template_id} not found! "
                           f"Available prompt ids are {list(self.templates_dict.keys())}")

        self.templates_dict = {force_template_id: self.__class__.templates_dict[force_template_id]}

        return self

    # function decorator needed to declare mandatory arguments of each subclass __call__
    @staticmethod
    def validate_args(*mandatory_args: str):
        def decorator(func):
            def wrapper(self, **kwargs):
                for mandatory_arg in mandatory_args:
                    assert mandatory_arg in kwargs, f"{mandatory_arg} is needed for task {repr(self)}!"

                return func(self, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def train(cls):
        Task.training = True

    @classmethod
    def eval(cls):
        Task.training = False

    @classmethod
    def from_string(cls, *task_str: str, all_unique_items: np.ndarray[str]):

        instantiated_tasks = []
        for task in task_str:
            try:
                # remember, we are searching a case-insensitive dict, so we don't care about
                # lowering all keys
                instantiated_tasks.append(cls.str_alias_cls[task](all_unique_items))
            except KeyError:
                raise KeyError(f"{task} task does not exist!") from None

        return instantiated_tasks

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return repr(self)
