from dataclasses import dataclass

from . import metrics
from .metrics import *

from src.evaluate.abstract_metric import Metric
from src.data.abstract_templates import Task


@dataclass
class EvalParams:
    metrics: tuple[str] = ("hit@10", "map@10", "mrr@10", "ndcg@10")
    eval_batch_size: int = None
    eval_tasks: tuple[str] = None
    create_latex_table: bool = True

    @classmethod
    def from_parse(cls, eval_section: dict):

        # the eval section does not have any nested structure,
        # thus for now complex parsing is not needed
        obj = cls(**eval_section)

        # check that each metric exists
        for metric_name in obj.metrics:
            Metric.metric_exists(metric_name, raise_error=True)

        # check that each eval task exists
        if obj.eval_tasks is not None:
            for task_name in obj.eval_tasks:
                Task.task_exists(task_name, raise_error=True)

        return obj
