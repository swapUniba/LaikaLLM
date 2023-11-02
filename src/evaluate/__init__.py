from dataclasses import dataclass

from . import metrics
from .metrics import *


@dataclass
class EvalParams:
    metrics: tuple[str] = ("hit@10", "map@10", "mrr@10")
    eval_batch_size: int = None
    eval_tasks: tuple[str] = None
    create_latex_table: bool = True

    @classmethod
    def from_parse(cls, eval_section: dict):

        # the eval section does not have any nested structure,
        # thus for now complex parsing is not needed
        return cls(**eval_section)
