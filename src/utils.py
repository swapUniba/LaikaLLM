import os
import random
from contextlib import contextmanager
from typing import List, Dict

import numpy as np
import torch
import torch.backends.cudnn
import wandb
import yaml
from cytoolz import merge_with
from yaspin import yaspin
from yaspin.spinners import Spinners


def seed_everything(seed: int):
    """
    Function which fixes the random state of each library used by this repository with the seed
    specified when invoking `pipeline.py`

    Returns:
        The integer random state set via command line argument

    """

    # seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    return seed


def log_wandb(parameters_to_log: dict, should_log: bool):
    if should_log is True:
        wandb.log(parameters_to_log)


@contextmanager
def init_wandb(should_log: bool, **kwargs):
    if should_log is True:
        project = kwargs.pop("project", "P5-Thesis")
        exp_name = kwargs.pop("name", None)

        with wandb.init(project=project, name=exp_name, **kwargs):
            yield
    else:
        yield


def list_dict2dict_list(list_of_dicts: List[dict]) -> Dict[str, list]:
    return merge_with(list, *list_of_dicts)


def dict_list2list_dict(dict_of_lists: Dict[str, list]) -> List[dict]:
    return [dict(zip(dict_of_lists, vals)) for vals in zip(*dict_of_lists.values())]


class IndentedDumper(yaml.Dumper):

    # this dumper indents also sequences other than mappings
    def increase_indent(self, flow=False, *args, **kwargs):
        return super().increase_indent(flow=flow, indentless=False)


class PrintWithSpin:

    def __init__(self, text: str):
        self.text = f"# {text}:"
        self.yaspin_obj = None

    def __enter__(self):

        self.yaspin_obj = yaspin(Spinners.sand, text=self.text, side="right").__enter__()

    def __exit__(self, exc_type, exc_value, traceback):

        self.yaspin_obj.ok("✔ Done!")


def format_time(seconds):
    # Convert seconds to minutes and seconds
    minutes, seconds = divmod(seconds, 60)

    # Convert minutes to hours and minutes
    hours, minutes = divmod(minutes, 60)

    # Format the time as a string
    if hours > 0:
        return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
    elif minutes > 0:
        return f"{int(minutes)} minutes, {int(seconds)} seconds"
    else:
        return f"{int(seconds)} seconds"
