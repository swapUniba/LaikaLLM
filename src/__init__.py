import os
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# format logging for a more user-friendly approach
logger.remove(0)
logger.add(sys.stderr, format="<level>{level}</level>: <level>{message}</level>", colorize=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = str(Path(os.path.join(_THIS_DIR, "..")).resolve())

DATA_DIR = os.path.join(ROOT_PATH, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_PATH, "models")
REPORTS_DIR = os.path.join(ROOT_PATH, "reports")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")


@dataclass
class GeneralParams:
    exp_name: str
    device: str = "cuda:0"
    random_seed: int = 42
    log_wandb: bool = False
    wandb_project: str = None
    eval_only: bool = False

    @classmethod
    def from_parse(cls, general_section):

        return cls(**general_section)
