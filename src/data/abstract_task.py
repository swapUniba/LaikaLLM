from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from requests.structures import CaseInsensitiveDict

if TYPE_CHECKING:
    from src.evaluate.abstract_metric import LaikaMetric


class LaikaTask(ABC):

    # keys are integers or str, values are Template objects
    templates_dict: dict[int | str, Template] = {}

    # name obj class mapping, used for when task must be initialized from strings
    str_alias_cls: dict[str, type[LaikaTask]] = CaseInsensitiveDict()

    # class attribute since if the model is in training mode, all tasks should be in training mode
    training: bool = False

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):

        if not inspect.isabstract(cls):
            cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    @classmethod
    @abstractmethod
    def compatible_metrics(cls) -> list[type[LaikaMetric]]:
        # all metrics which can be used to evaluate the results of the task
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_ranking_task(cls) -> bool:
        raise NotImplementedError

    def all_templates(self, return_id: bool = False):
        return list(self.templates_dict.keys()) if return_id else list(self.templates_dict.values())

    @abstractmethod
    def inference_templates(self, return_id: bool = False):
        raise NotImplementedError

    def force_template(self, force_template_id: int | str):

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
        LaikaTask.training = True

    @classmethod
    def eval(cls):
        LaikaTask.training = False

    @classmethod
    def all_tasks_available(cls, return_str: bool = False):
        return list(cls.str_alias_cls.keys()) if return_str else list(cls.str_alias_cls.values())

    @classmethod
    def task_exists(cls, task_cls_name: str, template_id: int | str = None,
                    return_bool: bool = True) -> bool | type[LaikaTask]:

        try:
            task_cls = cls.str_alias_cls[task_cls_name]
        except KeyError:
            raise KeyError(f"LaikaTask {task_cls_name} does not exist!") from None

        if template_id is not None and template_id not in task_cls.templates_dict.keys():
            raise KeyError(f"Template {template_id} for task {task_cls_name} does not exist!") from None

        return task_cls if not return_bool else True

    @classmethod
    def from_string(cls, task_str: str):

        task_cls = cls.task_exists(task_cls_name=task_str, return_bool=False)

        return task_cls()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> list[TaskOutput]:
        raise NotImplementedError

    def __eq__(self, other):
        if type(self) == type(other) and self.templates_dict == other.templates_dict:
            return True
        return False

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(str(self))


@dataclass
class TaskOutput:
    input_text: str
    target_text: str
    ground_truth_for_eval: list[str] = None

    # iter just so that this class can be unpacked,
    # e.g. input_prompt, target_text, gt = TaskOutput(...)
    def __iter__(self):
        return iter((self.input_text, self.target_text, self.ground_truth_for_eval))


@dataclass
class Template:
    input_text_placeholder: str
    target_text_placeholder: str

    # iter just so that this class can be unpacked,
    # e.g. input_prompt, target_text = Template(...)
    def __iter__(self):
        return iter((self.input_text_placeholder, self.target_text_placeholder))
