from __future__ import annotations

import operator
from typing import Callable

import numpy as np

from src.evaluate.abstract_metric import LaikaMetric, PaddedArr


class RankingMetric(LaikaMetric):

    @property
    def operator_comparison(self):
        return operator.gt

    def per_user_precomputed_matrix(self, predictions: np.ndarray[np.ndarray[str]], truths: PaddedArr):

        # If K is none a new dimension is added! Important to be sure k is not None
        if self.k is not None:
            predictions = predictions[:, :self.k]

        # no need to check if preds are != <PAD> to avoid that <PAD> tokens in pred and truth match,
        # since predictions are surely not padded
        result = predictions[:, np.newaxis, :] == truths[:, :, np.newaxis]

        rel_matrix = result.any(axis=1)

        return rel_matrix

    def __eq__(self, other):
        if isinstance(other, self.__class__) and other.k == self.k:
            return True
        return False

    def __str__(self):
        string = self.__class__.__name__
        if self.k is not None:
            string += f"@{self.k}"

        return string


class Hit(RankingMetric):

    def __call__(self, per_user_per_user_precomputed_matrix: np.ndarray) -> float:

        # intuitively, we want to """remove""" the dimension of items (axis=1) and maintain
        # the user dimension (axis=0). This variable will contain a bool value for each user: if at least one prediction
        # is relevant (appears in the user ground truth) for the user, the bool value is True
        per_user_hit = np.any(per_user_per_user_precomputed_matrix, axis=1)

        return np.mean(per_user_hit).item()


class MAP(RankingMetric):

    def __call__(self, per_user_precomputed_matrix: np.ndarray) -> float:

        cumulative_tp_matrix = np.cumsum(per_user_precomputed_matrix, axis=1)

        pos_arr = np.arange(1, per_user_precomputed_matrix.shape[1] + 1)
        position_rel_preds = pos_arr * per_user_precomputed_matrix  # it works thanks to numpy broadcasting

        cumulative_precision = self.safe_div(cumulative_tp_matrix, position_rel_preds)
        rel_count_per_user = np.count_nonzero(per_user_precomputed_matrix, axis=1)

        aps = self.safe_div(cumulative_precision.sum(axis=1), rel_count_per_user)

        map = np.mean(aps).item()

        return map


class MRR(RankingMetric):

    def __call__(self, per_user_precomputed_matrix: np.ndarray) -> float:

        # valid_preds are those predictions for which at least one relevant item has been recommended
        valid_preds = per_user_precomputed_matrix.any(axis=1)

        # take from valid preds the first occurrence of a relevant item (+1 since arrays start from 0)
        first_occ_rel_pred = per_user_precomputed_matrix[valid_preds].argmax(axis=1) + 1

        # compute rr for all users that have at least one rel item in their rec list
        valid_rrs = 1 / first_occ_rel_pred

        # predictions.shape[0] so that we take into account also users for which
        # no rel item has been recommended, effectively considering their rr == 0
        mrr = valid_rrs.sum() / per_user_precomputed_matrix.shape[0]

        return mrr


class NDCG(RankingMetric):

    # no different gains option because atm relevance is binary
    # thus it won't make a difference if gains were "linear" or "exp"
    def __init__(self, k: int = None, discount_log: Callable = np.log2):

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

    def __call__(self, per_user_precomputed_matrix: np.ndarray[np.ndarray[bool]]):

        per_user_precomputed_matrix = per_user_precomputed_matrix.astype(int)

        ndcgs = self._calc_ndcg(per_user_precomputed_matrix)

        return ndcgs.mean().item()
