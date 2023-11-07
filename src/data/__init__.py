from __future__ import annotations
from dataclasses import dataclass

from .datasets import *
from .templates import *

from src.data.abstract_dataset import LaikaDataset


@dataclass
class DataParams:
    dataset_cls_name: str
    dataset_params: dict

    @classmethod
    def from_parse(cls, data_section: dict):

        assert len(data_section) == 1, "Data section should specify only one dataset to use for the experiment!"

        dataset_name = list(data_section.keys())[0]
        dataset_params = data_section[dataset_name]

        obj = cls(dataset_name, dataset_params)

        # check that string params are valid
        LaikaDataset.dataset_exists(dataset_cls_name=obj.dataset_cls_name, raise_error=True)

        return obj
