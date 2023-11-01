from dataclasses import dataclass

from . import models
from .models import *

from src.data.abstract_templates import Task
from src.model.abstract_model import LaikaModel


@dataclass
class ModelParams:

    # model params
    model_cls_name: str
    model_kwargs: dict

    # trainer params
    n_epochs: int = 10
    train_tasks: tuple[str] = tuple(Task.str_alias_cls.keys())
    val_task: str = None
    val_task_template_id: str = None
    monitor_strategy: str = None
    train_batch_size: int = 4
    eval_batch_size: int = train_batch_size

    @classmethod
    def from_parse(cls, model_section: dict):

        valid_model_found = set(model_section.keys()).intersection(LaikaModel.str_alias_cls.keys())

        if len(valid_model_found) == 0:
            raise ValueError("Missing model class from the 'model' section!")
        if len(valid_model_found) > 1:
            raise ValueError(f"Only one model expected, found {list(valid_model_found)}")

        # we are sure there is only one element from the above checks, so
        # we are popping the valid model name
        model_name = valid_model_found.pop()
        model_kwargs = model_section[model_name]

        # pop so that we can forward all the model section to the dataclass __init__,
        # as it will contain only the trainer params
        model_section.pop(model_name)

        return cls(model_cls_name=model_name, model_kwargs=model_kwargs, **model_section)
