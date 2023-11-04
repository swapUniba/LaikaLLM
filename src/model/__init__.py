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
    val_task_template_id: int | str = None

    # trainer params
    n_epochs: int = 10
    monitor_metric: str = "loss"
    train_batch_size: int = 4
    eval_batch_size: int = train_batch_size

    @classmethod
    def from_parse(cls, model_section: dict):

        # model specification should be the first subsection of the model section
        model_name = list(model_section.keys())[0]

        if model_name in cls.__annotations__.keys():
            raise ValueError(f"Found {model_name} as first element of the 'model' section, "
                             f"but the model definition with its params it's expected!")

        model_kwargs = model_section[model_name]

        # pop so that we can forward all the model section to the dataclass __init__,
        # as it will contain only the trainer params
        model_section.pop(model_name)

        obj = cls(model_cls_name=model_name, model_kwargs=model_kwargs, **model_section)

        # check that model exists
        LaikaModel.model_exists(obj.model_cls_name, raise_error=True)

        # check that each train task exist
        for task_name in obj.train_tasks:
            Task.task_exists(task_name, raise_error=True)

        # check that valid task and template exist
        if obj.val_task is not None:
            Task.task_exists(obj.val_task, template_id=obj.val_task_template_id, raise_error=True)

        # check that monitor metric exists
        Metric.metric_exists(obj.monitor_metric, raise_error=True)

        return obj
