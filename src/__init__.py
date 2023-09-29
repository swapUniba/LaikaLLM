import os
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = str(Path(os.path.join(THIS_DIR, "..")).resolve())

DATA_DIR = os.path.join(ROOT_PATH, "data")
SNAP_DIR = os.path.join(ROOT_PATH, "snap")
MODELS_DIR = os.path.join(ROOT_PATH, "models")


@dataclass
class ExperimentConfig:

    exp_name: str = None
    checkpoint: str = "google/flan-t5-small"
    n_epochs: int = 10
    integer_ids: bool = False
    inject_personalization: bool = False
    train_batch_size: int = 4
    eval_batch_size: int = 2
    device: str = "cuda:0"
    random_seed: int = 42
    log_wandb: bool = False
