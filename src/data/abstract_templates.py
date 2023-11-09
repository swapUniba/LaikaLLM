from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from requests.structures import CaseInsensitiveDict

if TYPE_CHECKING:
    from src.evaluate.abstract_metric import LaikaMetric


class PromptTarget:

    def __init__(self, input_prompt: str, target_text: str, gt: list[str] = None):
        self.input_prompt = input_prompt
        self.target_text = target_text
        self.gt = gt

    # iter just so that this class can be unpacked,
    # e.g. input_prompt, target_text, gt = PromptTarget(...)
    def __iter__(self):
        return iter((self.input_prompt, self.target_text, self.gt))

    def __str__(self):
        string = " Input ".center(50, "#") + "\n"
        string += self.input_prompt + "\n"
        string += " Target ".center(50, "#") + "\n"
        string += self.target_text

        return string


class Task(ABC):

    # keys are integers or str, values are PromptTarget objects
    templates_dict: dict[int | str, PromptTarget] = {}

    # all metrics which can be used to evaluate the results of the task
    compatible_metrics: list[type[LaikaMetric]] = []

    # name obj class mapping, used for when task must be initialized from strings
    str_alias_cls: dict[str, type[Task]] = CaseInsensitiveDict()

    # class attribute since if the model is in training mode, all tasks should be in training mode
    training: bool = False

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):

        if not inspect.isabstract(cls):
            cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    @property
    @abstractmethod
    def is_ranking_task(self) -> bool:
        raise NotImplementedError

    def all_templates(self, return_id: bool = False):
        return list(self.templates_dict.keys()) if return_id else list(self.templates_dict.values())

    @abstractmethod
    def inference_templates(self, return_id: bool = False):
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

    @classmethod
    def all_tasks_available(cls, return_str: bool = False):
        return list(cls.str_alias_cls.values()) if return_str else list(cls.str_alias_cls.keys())

    @classmethod
    def task_exists(cls, task_cls_name: str, template_id: int | str = None, raise_error: bool = True):

        # if no template id specified, then it should not count towards the result,
        # hence set to True (True is neutral element of AND operation)
        template_exists = True
        task_exists = task_cls_name in cls.str_alias_cls.keys()

        if task_exists and template_id is not None:
            template_exists = template_id in cls.str_alias_cls[task_cls_name].templates_dict.keys()

            if not template_exists and raise_error is True:
                raise KeyError(f"Template {template_id} for task {task_cls_name} does not exist!")

        if not task_exists and raise_error is True:
            raise KeyError(f"Task {task_cls_name} does not exist!")

        return task_exists and template_exists

    @abstractmethod
    def __call__(self, *args, **kwargs) -> list[tuple[str, str]]:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(str(self))
