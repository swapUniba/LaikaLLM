from __future__ import annotations

import inspect
from abc import abstractmethod, ABC
from typing import List

import numpy as np
import torch
from requests.structures import CaseInsensitiveDict

from src.data.abstract_dataset import LaikaDataset
from src.data.abstract_templates import Task


class LaikaModel(ABC):
    str_alias_cls: dict[str, type[LaikaModel]] = CaseInsensitiveDict()

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):
        if not inspect.isabstract(cls):
            cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    def __init__(self, training_tasks_str: List[str],
                 all_unique_labels: List[str],
                 eval_task_str: str = None,
                 eval_template_id: int | str = None):

        if training_tasks_str is None:
            raise AttributeError("training_tasks_str parameter can't be None!")
        if all_unique_labels is None:
            raise AttributeError("all_unique_labels parameter can't be None!")

        self.all_unique_labels = np.array(all_unique_labels)
        self.training_tasks = Task.from_string(*training_tasks_str,
                                               all_unique_items=self.all_unique_labels)

        self.eval_task = None
        if eval_task_str is not None:
            self.set_eval_task(eval_task_str, eval_template_id)

    def set_eval_task(self, eval_task_str: str, template_id: int = None):
        [self.eval_task] = Task.from_string(eval_task_str, all_unique_items=self.all_unique_labels)

        if template_id is not None:
            self.eval_task.force_template_id(template_id)

    @property
    @abstractmethod
    def get_suggested_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, batch: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def prepare_input(self, tokenized_batch: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, prepared_batch: dict) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    # if labels are in "prepared_batch", also valid loss should be returned
    def generate_step(self, prepared_batch: dict) -> tuple[torch.FloatTensor, np.ndarray[str]] | np.ndarray[str]:
        raise NotImplementedError

    @abstractmethod
    def train(self, mode: bool = True):
        raise NotImplementedError

    def eval(self):
        Task.eval()

        return self.train(False)

    @abstractmethod
    def save(self, output_dir: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str, **kwargs) -> LaikaModel:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: str):
        raise NotImplementedError

    @classmethod
    def from_cls(cls, model_cls: type[LaikaModel], dataset_obj: LaikaDataset, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_string(cls, model_cls_name: str, dataset_obj: LaikaDataset, **kwargs):

        try:
            model_cls = cls.str_alias_cls[model_cls_name]
        except KeyError:
            raise KeyError(f"Model {model_cls_name} does not exist!") from None

        # it seems a recursive call, but the top level (LaikaModel) is an abstract class,
        # we are basically calling the from_string of the subclass
        return model_cls.from_cls(model_cls, dataset_obj, **kwargs)

    @classmethod
    def all_models_available(cls, return_str: bool = False):
        return list(cls.str_alias_cls.values()) if return_str else list(cls.str_alias_cls.keys())

    @classmethod
    def model_exists(cls, model_cls_name: str, raise_error: bool = True):

        model_exists = model_cls_name in cls.str_alias_cls.keys()

        if not model_exists and raise_error is True:
            raise KeyError(f"Metric {model_cls_name} does not exist!")

        return model_exists
