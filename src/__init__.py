import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Tuple, Literal

from src.data.templates import Task

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = str(Path(os.path.join(THIS_DIR, "..")).resolve())

DATA_DIR = os.path.join(ROOT_PATH, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SNAP_DIR = os.path.join(ROOT_PATH, "snap")
MODELS_DIR = os.path.join(ROOT_PATH, "models")


@dataclass
class ExperimentConfig:

    exp_name: str = None
    checkpoint: str = "google/flan-t5-small"
    n_epochs: int = 10
    train_tasks: Tuple[str] = tuple(Task.str_alias_obj.keys())
    integer_ids: bool = False
    items_start_from_1001: bool = False
    inject_personalization: Tuple[str] = ()
    monitor_strategy: Literal['no', 'loss', 'hit@10'] = "no"
    train_batch_size: int = 4
    eval_batch_size: int = 2
    add_prefix_items_users: bool = False
    device: str = "cuda:0"
    random_seed: int = 42
    log_wandb: bool = False

    @classmethod
    def to_dict(cls):
        return {field: getattr(cls, field) for field in cls.__annotations__}

    @classmethod
    def to_string(cls):
        return pprint(cls.to_dict())
