import argparse
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

from src import ExperimentConfig


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
    print(f"Random seed set as {seed}")

    return seed


def log_wandb(parameters_to_log: dict):
    if ExperimentConfig.log_wandb:
        wandb.log(parameters_to_log)


@contextmanager
def init_wandb(**kwargs):
    if ExperimentConfig.log_wandb:
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


class LoadFromYaml(argparse.Action):

    def __init__(self, nargs='?', **kw):
        super().__init__(nargs=nargs, **kw)

    def _update_namespace(self, namespace, param_section_dict: dict):
        for param_name, param_val in param_section_dict.items():
            # set arguments in the target namespace if they exist, otherwise raise error
            if hasattr(namespace, param_name) is True:
                setattr(namespace, param_name, param_val)
            else:
                raise argparse.ArgumentError(self, f"Unrecognized argument read from yaml config -> {param_name}")

    def __call__(self, parser, namespace, values, option_string=None):
        with open(values, "r") as f:
            yaml_args = yaml.safe_load(f)

        data_section = yaml_args.pop("data", None)
        model_section = yaml_args.pop("model", None)
        eval_section = yaml_args.pop("eval", None)

        # after popping every section, only general params remain
        general_section = yaml_args

        if general_section is not None:
            self._update_namespace(namespace, general_section)

        if data_section is not None:
            self._update_namespace(namespace, data_section)

        if model_section is not None:
            self._update_namespace(namespace, model_section)

        if eval_section is not None:
            self._update_namespace(namespace, eval_section)
