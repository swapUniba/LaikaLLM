from __future__ import annotations

import inspect
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

        if not inspect.isabstract(cls):
            cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    def __init__(self):

        # the common process to all dataset is surely to download and extract raw data
        self.download_extract_raw_dataset()

    @property
    @abstractmethod
    def all_users(self) -> np.ndarray[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def all_items(self) -> np.ndarray[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def items_meta_dict(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def download_extract_raw_dataset(self):
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
    def all_datasets_available(cls, return_str: bool = False) -> list[type[LaikaDataset] | str]:
        return list(cls.str_alias_cls.keys()) if return_str else list(cls.str_alias_cls.values())

    @classmethod
    def dataset_exists(cls, dataset_cls_name: str, return_bool: bool = True) -> bool | type[LaikaDataset]:

        try:
            dataset_cls = cls.str_alias_cls[dataset_cls_name]
        except KeyError:
            raise KeyError(f"Dataset {dataset_cls_name} does not exist!") from None

        # if we arrive at the return clause, dataset_cls exists that's why we return True directly
        return dataset_cls if not return_bool else True

    @classmethod
    def from_string(cls, dataset_cls_name: str, **dataset_params):

        dataset_cls = cls.dataset_exists(dataset_cls_name, return_bool=False)

        # wrong warning, subclasses may have parameters in __init__ method,
        # and in any case dataset_params can be empty, in that case __init__ will take
        # no parameter
        return dataset_cls(**dataset_params)  # type: ignore
