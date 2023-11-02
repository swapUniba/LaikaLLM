from dataclasses import dataclass

from . import metrics
from .metrics import *


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

        # normalize string params to lower
        obj.metrics = tuple(metric_name.lower() for metric_name in obj.metrics)

        if obj.eval_tasks is not None:
            obj.eval_tasks = tuple(eval_task_name.lower() for eval_task_name in obj.eval_tasks)

        return obj
