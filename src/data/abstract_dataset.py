from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import datasets
import numpy as np
import pandas as pd
from requests.structures import CaseInsensitiveDict


class LaikaDataset(ABC):

    str_alias_cls: dict[str, type[LaikaDataset]] = CaseInsensitiveDict()

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):
        cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    @property
    @abstractmethod
    def all_users(self) -> np.ndarray[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def all_items(self) -> np.ndarray[str]:
        raise NotImplementedError

    @abstractmethod
    def split_data(self, original_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    # important that this is a static method, otherwise slow hashing for map fn of huggingface!
    def sample_train_sequence(batch: Dict[str, list]) -> Dict[str, list]:
        raise NotImplementedError

    @abstractmethod
    def get_hf_datasets(self, merge_train_val: bool = False) -> Dict[str, datasets.Dataset]:
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str) -> LaikaDataset:
        raise NotImplementedError

    @classmethod
    def from_string(cls, dataset_cls_name: str, **dataset_params):

        try:
            dataset_cls = cls.str_alias_cls[dataset_cls_name]
        except KeyError:
            raise KeyError(f"Dataset {dataset_cls_name} does not exist!") from None

        return dataset_cls(**dataset_params)
