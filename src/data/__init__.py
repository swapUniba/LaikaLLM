from dataclasses import dataclass

from . import datasets
from .datasets import *


@dataclass
class DataParams:
    dataset_cls_name: str
    dataset_params: dict

    @classmethod
    def from_parse(cls, data_section: dict):

        assert len(data_section) == 1, "Data section should specify only one dataset to use for the experiment!"

        dataset_name = list(data_section.keys())[0]
        dataset_params = data_section[dataset_name]

        return cls(dataset_name, dataset_params)
