from dataclasses import dataclass

from . import models
from .models import *

from src.model.abstract_model import LaikaModel


@dataclass
class ModelParams:

    # model params
    model_cls_name: str
    model_kwargs: dict
    train_tasks: tuple[str]
    val_task: str = None
    val_task_template_id: str = None

    # trainer params
    n_epochs: int = 10
    monitor_metric: str = "loss"
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

        obj = cls(model_cls_name=model_name, model_kwargs=model_kwargs, **model_section)

        # normalize strings params to lower
        obj.train_tasks = tuple(train_task_name.lower() for train_task_name in obj.train_tasks)
        obj.val_task = obj.val_task.lower() if obj.val_task is not None else None
        obj.monitor_metric = obj.monitor_metric.lower()

        return obj
