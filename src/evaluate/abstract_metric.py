from __future__ import annotations

import inspect
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
    def operator_comparison(self):

        # What is the operator to use if we want to obtain the best result the metric?
        # e.g. loss is "<", hit is ">", mse is "<", etc.
        # By default is ">"
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def per_user_precomputed_matrix(predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[np.ndarray[str]],
                                    **kwargs):
        raise NotImplementedError

    @staticmethod
    def safe_div(num: np.ndarray, den: np.ndarray):

        # divide only if denominator is different from 0, otherwise 0
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)

    @classmethod
    def from_string(cls, *metric_str: str) -> list[LaikaMetric]:

        # this should be improved: make use of subclasses polymorphism to convert from string to object

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

                    # wrong warning, ErrorMetrics don't have k, but this is expected behaviour,
                    # if the user sets for example mae@5 obviously it doesn't make sense and an error is raised
                    instantiated_metrics.append(cls.str_alias_cls[metric_name](k=int(k)))

            except KeyError:
                raise KeyError(f"{metric} metric does not exist!") from None

        return instantiated_metrics

    @classmethod
    def all_metrics_available(cls, return_str: bool = False):
        return list(cls.str_alias_cls.values()) if return_str else list(cls.str_alias_cls.keys())

    @classmethod
    def metric_exists(cls, metric_cls_name: str, raise_error: bool = True):

        # this should be improved: make use of subclasses polymorphism to convert from string to object

        # regardless if there is the cutoff value k or not, we are only interested in the metric name
        # which is the part before the optional '@' symbol
        metric_exists = metric_cls_name.split("@")[0] in cls.str_alias_cls.keys()

        if not metric_exists and raise_error is True:
            raise KeyError(f"Metric {metric_cls_name} does not exist!")

        return metric_exists

    @abstractmethod
    def __call__(self, precomputed_matrix: np.ndarray) -> float:
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return True
        return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        string = self.__class__.__name__

        return string


class Loss(LaikaMetric):

    @property
    def operator_comparison(self):
        # loss metric should be minimized, hence "<"
        return operator.lt

    @staticmethod
    def per_user_precomputed_matrix(predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[np.ndarray[str]],
                                    **kwargs):
        raise NotImplementedError("This should not be called, it is simply defined to make use of polymorphism")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This should not be called, it is simply defined to make use of polymorphism")
