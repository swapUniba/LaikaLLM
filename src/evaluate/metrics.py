from __future__ import annotations

import itertools
import math
import operator
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Callable

import numpy as np
from requests.structures import CaseInsensitiveDict


class PaddedArr(np.ndarray):
    def __new__(cls, iterable: Collection[Collection[str]], *args, **kwargs):
        # Find the maximum length of the sublists
        max_len = max(len(sublist) for sublist in iterable)

        # Create a new NumPy array filled with <PAD> token
        padded_array = np.full((len(iterable), max_len), fill_value="<PAD>")

        # Copy the data from the original list into the new array
        for i, sublist in enumerate(iterable):
            padded_array[i, :len(sublist)] = sublist

        return padded_array.astype(str)


class Metric(ABC):

    # name - class mapping, used for when metrics should be initialized from strings
    str_alias_cls: dict = CaseInsensitiveDict()

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):
        cls.str_alias_cls[cls.__name__] = cls

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

        predictions = predictions[:, k]

        result = (predictions[:, np.newaxis, :] == truths[:, :, np.newaxis]) & \
                 (predictions[:, np.newaxis, :] != "<PAD>")

        rel_matrix = result.any(axis=1)

        return rel_matrix

    @staticmethod
    def safe_div(num: np.ndarray, den: np.ndarray):

        # divide only if denominator is different from 0, otherwise 0
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)

    @classmethod
    def from_string(cls, *metric_str: str) -> List[Metric]:

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

    @abstractmethod
    def __call__(self, rel_binary_matrix: np.ndarray[np.ndarray[bool]]) -> float:
        raise NotImplementedError

    def __str__(self):
        string = self.__class__.__name__
        if self.k is not None:
            string += f"@{self.k}"

        return string


class Hit(Metric):

    def __call__(self, rel_binary_matrix: np.ndarray[np.ndarray[bool]]) -> float:

        # intuitively, we want to """remove""" the dimension of items (axis=1) and maintain
        # the user dimension (axis=0). This variable will contain a bool value for each user: if at least one prediction
        # is relevant (appears in the user ground truth) for the user, the bool value is True
        per_user_hit = np.any(rel_binary_matrix, axis=1)

        return np.mean(per_user_hit).item()


class MAP(Metric):

    def __call__(self, rel_binary_matrix: np.ndarray[np.ndarray[bool]]) -> float:

        cumulative_tp_matrix = np.cumsum(rel_binary_matrix, axis=1)

        pos_arr = np.arange(1, rel_binary_matrix.shape[1] + 1)
        position_rel_preds = pos_arr * rel_binary_matrix  # it works thanks to numpy broadcasting

        cumulative_precision = self.safe_div(cumulative_tp_matrix, position_rel_preds)
        rel_count_per_user = np.count_nonzero(rel_binary_matrix, axis=1)

        aps = self.safe_div(cumulative_precision.sum(axis=1), rel_count_per_user)

        map = np.mean(aps).item()

        return map


class MRR(Metric):

    def __call__(self, rel_binary_matrix: np.ndarray[np.ndarray[bool]]) -> float:

        # valid_preds are those predictions for which at least one relevant item has been recommended
        valid_preds = rel_binary_matrix.any(axis=1)

        # take from valid preds the first occurrence of a relevant item (+1 since arrays start from 0)
        first_occ_rel_pred = rel_binary_matrix[valid_preds].argmax(axis=1) + 1

        # compute rr for all users that have at least one rel item in their rec list
        valid_rrs = 1 / first_occ_rel_pred

        # predictions.shape[0] so that we take into account also users for which
        # no rel item has been recommended, effectively considering their rr == 0
        mrr = valid_rrs.sum() / rel_binary_matrix.shape[0]

        return mrr


class NDCG(Metric):

    # no different gains option because atm relevance is binary
    # thus it won't make a difference if gains were "linear" or "exp"
    def __init__(self,
                 k: int = None,
                 discount_log: Callable = np.log2):

        super().__init__(k)

        self.discount_log = discount_log

    def _dcg_score(self, r: np.ndarray[np.ndarray]) -> np.ndarray[float]:
        """Discounted cumulative gain (DCG)
        Parameters
        ----------
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        Returns
        -------
        DCG : float
        """

        discounts = self.discount_log(np.arange(2, r.shape[1] + 2))

        # division possible thanks to numpy broadcasting
        # safe division not necessary, arange starts from 2 and can't be 0
        dcg = np.sum(r / discounts, axis=1)

        return dcg

    def _calc_ndcg(self, r: np.ndarray[np.ndarray]) -> np.ndarray[float]:
        """Normalized discounted cumulative gain (NDCG)
        Parameters
        ----------
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        Returns
        -------
        NDCG : float
        """
        actual_ndcgs = self._dcg_score(r)

        # Get the indices that would sort each row in descending order
        sorted_indices = np.argsort(-r, axis=1)

        # Use the sorted_indices to sort the matrix row-wise in descending order
        ideal_predictions = r[np.arange(r.shape[0])[:, np.newaxis], sorted_indices]

        ideal_ndcgs = self._dcg_score(ideal_predictions)

        return self.safe_div(actual_ndcgs, ideal_ndcgs)

    def __call__(self, rel_binary_matrix: np.ndarray[np.ndarray[bool]]):

        rel_binary_matrix = rel_binary_matrix.astype(int)

        ndcgs = self._calc_ndcg(rel_binary_matrix)

        return ndcgs.mean().item()


        return np.mean(ndcgs).item()
