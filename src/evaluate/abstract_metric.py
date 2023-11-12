from __future__ import annotations

import inspect
from abc import abstractmethod, ABC
import operator

from requests.structures import CaseInsensitiveDict
from typing import Collection, Callable

import numpy as np


class PaddedArr(np.ndarray):
    def __new__(cls, iterable: Collection[Collection[str]], *args, **kwargs):
        # Find the maximum length of the sublists
        max_len = max(len(sublist) for sublist in iterable)

        # Create a new NumPy array filled with <PAD> token
        # object dtype here since using `str` will use the numpy str type based on <PAD> string
        # cutting strings significantly longer than <PAD>
        padded_array = np.full((len(iterable), max_len), fill_value="<PAD>", dtype=object)

        # Copy the data from the original list into the new padded array
        for i, sublist in enumerate(iterable):
            padded_array[i, :len(sublist)] = sublist

            assert not (padded_array[i, :len(sublist)] == "<PAD>").any(), \
                "<PAD> is the pad token and can't be used as element of array!"

        return padded_array.astype(str)


class LaikaMetric(ABC):

    # name - class mapping, used for when metrics should be initialized from strings
    str_alias_cls: dict[str, type[LaikaMetric]] = CaseInsensitiveDict()

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):

        if not inspect.isabstract(cls):
            cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    def __init__(self, k: int = None):
        self.k = k

    @property
    @abstractmethod
    def operator_comparison(self) -> Callable:

        # What is the operator to use if we want to obtain the best result the metric?
        # e.g. loss is "<", hit is ">", mse is "<", etc.
        # By default is ">"
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def per_user_precomputed_matrix(predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[np.ndarray[str]],
                                    **kwargs) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:

        # divide only if denominator is different from 0, otherwise 0
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)

    @classmethod
    def from_string(cls, metric_str: str) -> LaikaMetric:

        # this should be improved: make use of subclasses polymorphism to convert from string to object
        try:
            metric_info = metric_str.split("@")

            match metric_info:

                case [metric_name]:
                    instantiated_metric = cls.str_alias_cls[metric_name]()

                case [metric_name, k]:
                    if not k.isdigit():
                        raise KeyError

                    instantiated_metric = cls.str_alias_cls[metric_name](k=int(k))

                case _:
                    raise KeyError

        except KeyError:
            raise KeyError(f"{metric_str} metric does not exist!") from None

        return instantiated_metric

    @classmethod
    def all_metrics_available(cls, return_str: bool = False) -> list[type[LaikaMetric] | str]:
        return list(cls.str_alias_cls.keys()) if return_str else list(cls.str_alias_cls.values())

    @classmethod
    def metric_exists(cls, metric_cls_name: str, return_bool: bool = True) -> bool | type[LaikaMetric]:

        # this should be improved: make use of subclasses polymorphism to convert from string to object

        # regardless if there is the cutoff value k or not, we are only interested in the metric name
        # which is the part before the optional '@' symbol
        metric_cls_name = metric_cls_name.split("@")[0]

        try:
            metric_cls = cls.str_alias_cls[metric_cls_name]
        except KeyError:
            raise KeyError(f"Metric {metric_cls_name} does not exist!") from None

        # if we arrive at the return clause, metric_cls exists that's why we return True directly
        return metric_cls if not return_bool else True

    @abstractmethod
    def __call__(self, precomputed_matrix: np.ndarray) -> float:
        raise NotImplementedError

    def __eq__(self, other):
        if type(self) == type(other) and self.k == other.k:
            return True
        return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        string = self.__class__.__name__

        return string


class Loss(LaikaMetric):

    @property
    def operator_comparison(self) -> Callable:
        # loss metric should be minimized, hence "<"
        return operator.lt

    @staticmethod
    def per_user_precomputed_matrix(predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[np.ndarray[str]],
                                    **kwargs):
        raise NotImplementedError("This should not be called, it is simply defined to make use of polymorphism")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This should not be called, it is simply defined to make use of polymorphism")
