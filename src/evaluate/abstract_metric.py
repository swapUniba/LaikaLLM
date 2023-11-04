from __future__ import annotations
from abc import abstractmethod, ABC
import operator

from requests.structures import CaseInsensitiveDict
from typing import Collection

import numpy as np


class PaddedArr(np.ndarray):
    def __new__(cls, iterable: Collection[Collection[str]], *args, **kwargs):
        # Find the maximum length of the sublists
        max_len = max(len(sublist) for sublist in iterable)

        # Create a new NumPy array filled with <PAD> token
        # object dtype here since using `str` will use the numpy str type based on <PAD> string
        # cutting strings greater than <PAD>
        padded_array = np.full((len(iterable), max_len), fill_value="<PAD>", dtype=object)

        # Copy the data from the original list into the new array
        for i, sublist in enumerate(iterable):
            padded_array[i, :len(sublist)] = sublist

        return padded_array.astype(str)


class Metric(ABC):

    # name - class mapping, used for when metrics should be initialized from strings
    str_alias_cls: dict[str, type[Metric]] = CaseInsensitiveDict()

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):
        cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    def __init__(self, k: int = None):
        self.k = k

    @property
    def operator_comparison(self):

        # What is the operator to use if we want to obtain the best result the metric?
        # e.g. loss is "<", hit is ">", mse is "<", etc.
        # By default is ">"
        return operator.gt

    @staticmethod
    def rel_binary_matrix(predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[np.ndarray[str]], k: int = None):

        # If K is none a new dimension is added! Important to be sure is != None
        if k is not None:
            predictions = predictions[:, :k]

        result = (predictions[:, np.newaxis, :] == truths[:, :, np.newaxis]) & \
                 (predictions[:, np.newaxis, :] != "<PAD>")

        rel_matrix = result.any(axis=1)

        return rel_matrix

    @staticmethod
    def safe_div(num: np.ndarray, den: np.ndarray):

        # divide only if denominator is different from 0, otherwise 0
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)

    @classmethod
    def from_string(cls, *metric_str: str) -> list[Metric]:

        instantiated_metrics = []
        for metric in metric_str:

            try:
                metric_info = metric.split("@")

                if len(metric_info) == 1:

                    [metric_name] = metric_info

                    instantiated_metrics.append(cls.str_alias_cls[metric_name]())
                elif len(metric_info) == 2:
                    [metric_name, k] = metric_info

                    if not k.isdigit():
                        raise KeyError

                    instantiated_metrics.append(cls.str_alias_cls[metric_name](k=int(k)))

            except KeyError:
                raise KeyError(f"{metric} metric does not exist!") from None

        return instantiated_metrics

    @classmethod
    def all_metrics_available(cls, return_str: bool = False):
        return list(cls.str_alias_cls.values()) if return_str else list(cls.str_alias_cls.keys())

    @classmethod
    def metric_exists(cls, metric_cls_name: str, raise_error: bool = True):

        # regardless if there is the cutoff value k or not, we are only interested in the metric name
        # which is the part before the '@' symbol
        metric_exists = metric_cls_name.split("@")[0] in cls.str_alias_cls.keys()

        if not metric_exists and raise_error is True:
            raise KeyError(f"Metric {metric_cls_name} does not exist!")

        return metric_exists

    @abstractmethod
    def __call__(self, rel_binary_matrix: np.ndarray[np.ndarray[bool]]) -> float:
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, self.__class__) and other.k == self.k:
            return True
        return False

    def __str__(self):
        string = self.__class__.__name__
        if self.k is not None:
            string += f"@{self.k}"

        return string
