from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torchmetrics.functional as torchmetrics_fn
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Metric(ABC):

    @abstractmethod
    def __call__(self, predictions: np.ndarray[np.ndarray[str] | str], truths: np.ndarray[str]) -> float:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class ClassificationMetric(Metric):

    @abstractmethod
    def compute_metric(self, predictions: torch.IntTensor, truths: torch.IntTensor, n_labels: int):
        raise NotImplementedError

    def __call__(self, predictions: np.ndarray[str], truths: np.ndarray[str]) -> float:
        total = np.hstack((predictions, truths))

        le = LabelEncoder().fit(total)
        predictions = torch.from_numpy(le.transform(predictions)).int()
        truths = torch.from_numpy(le.transform(truths)).int()

        num_classes = len(np.unique(total))

        res = self.compute_metric(predictions, truths, num_classes)

        # k is not used, there's no list to cut
        return res.item()

    def __str__(self):

        return f"{self.__class__.__name__} (weighted)"


class Accuracy(ClassificationMetric):

    def compute_metric(self, predictions: torch.IntTensor, truths: torch.IntTensor, n_labels: int):
        return torchmetrics_fn.classification.multiclass_accuracy(predictions,
                                                                  truths,
                                                                  num_classes=n_labels,
                                                                  average="weighted")


class Precision(ClassificationMetric):

    def compute_metric(self, predictions: torch.IntTensor, truths: torch.IntTensor, n_labels: int):
        return torchmetrics_fn.classification.multiclass_precision(predictions,
                                                                   truths,
                                                                   num_classes=n_labels,
                                                                   average="weighted")


class Recall(ClassificationMetric):

    def compute_metric(self, predictions: torch.IntTensor, truths: torch.IntTensor, n_labels: int):
        return torchmetrics_fn.classification.multiclass_recall(predictions,
                                                                truths,
                                                                num_classes=n_labels,
                                                                average="weighted")


class F1(ClassificationMetric):

    def compute_metric(self, predictions: torch.IntTensor, truths: torch.IntTensor, n_labels: int):
        return torchmetrics_fn.classification.multiclass_f1_score(predictions,
                                                                  truths,
                                                                  num_classes=n_labels,
                                                                  average="weighted")


class RankingMetric(Metric):

    def __init__(self, k: int = None):
        self.k = k

    def __str__(self):
        string = self.__class__.__name__
        if self.k is not None:
            string += f"@{self.k}"

        return string


class Hit(RankingMetric):

    def __call__(self, predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[str]) -> float:
        predictions = predictions[:, :self.k] if self.k is not None else predictions

        return np.mean(np.any(truths[:, np.newaxis] == predictions, axis=1)).item()


class MAP(RankingMetric):

    def __call__(self, predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[str]) -> float:

        predictions = predictions[:, :self.k] if self.k is not None else predictions

        aps = []

        for pred, truth in zip(predictions, truths):

            pred: np.ndarray
            truth: np.ndarray

            # all items both in prediction and relevant truth are retrieved, if an item only appears in prediction it is
            # marked with a -1, we then retrieve only the indexes of items that appear in both
            hits: np.ndarray = (pred == truth).nonzero()[0]

            if len(hits) == 0:
                aps.append(0)
                continue

            # we initialize an array for true positives. True positive is incremented by 1 each time a relevant item
            # is found, therefore this array will be as long as the array containing the indices of items both in
            # prediction and relevant truth (and values will be as such [1, 2, 3, ...])
            tp_array = np.arange(start=1, stop=len(hits) + 1)

            # finally, precision is computed by dividing each true positive value to each corresponding position
            precision_array = tp_array / (hits + 1)
            cumulative_precision = np.sum(precision_array)

            # in this case we only have one item in truth
            truth_ap = (1 / len(hits)) * cumulative_precision

            aps.append(truth_ap)

        map = np.mean(aps).item()

        return map


class MRR(RankingMetric):

    def __call__(self, predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[str]) -> float:

        predictions = predictions[:, :self.k] if self.k is not None else predictions

        rrs = []

        for pred, truth in zip(predictions, truths):

            pred: np.ndarray
            truth: np.ndarray

            rel_rank: np.ndarray = pred == truth
            non_zero_rel_rank = np.argwhere(rel_rank).flatten()

            if len(non_zero_rel_rank) > 0:
                # + 1 because the array starts from 0
                first_rel_rank = non_zero_rel_rank[0] + 1
                rr = 1 / first_rel_rank
                rrs.append(rr)
            else:
                rr = 0
                rrs.append(rr)

        mrrs = np.mean(rrs).item()
        return mrrs


class NDCG:
    def __init__(self,
                 k: int = None,
                 gains: Literal['exponential', 'linear'] = "linear",
                 discount_log: Callable = np.log2):

        if gains not in {"linear", "exponential"}:
            raise ValueError("Invalid gains option!")

        self.k = k
        self.gains = gains
        self.discount_log = discount_log

        if self.gains == "exponential":
            self.gains_fn = lambda r: 2 ** r - 1
        else:
            self.gains_fn = lambda r: r

    def _dcg_score(self, r: np.ndarray):
        """Discounted cumulative gain (DCG)
        Parameters
        ----------
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        Returns
        -------
        DCG : float
        """
        dcg = np.nan
        if len(r) != 0:

            gains = self.gains_fn(r)
            discounts = self.discount_log(np.arange(2, len(r) + 2))

            dcg = np.sum(gains / discounts)

        return dcg

    def _calc_ndcg(self, r: np.ndarray):
        """Normalized discounted cumulative gain (NDCG)
        Parameters
        ----------
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        Returns
        -------
        NDCG : float
        """
        actual = self._dcg_score(r)
        ideal = self._dcg_score(np.sort(r)[::-1])
        return actual / ideal

    def __call__(self, predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[str]):

        predictions = predictions[:, :self.k] if self.k is not None else predictions

        ndcgs = []
        for pred, truth in zip(predictions, truths):

            pred: np.ndarray
            truth: str

            rel_rank_mask = (pred == truth).astype(int)

            # scores is decreasing, since items in first positions
            # should have higher score. Relevance score starts from 1 rather than 0
            # e.g.
            # rel_rank_mask = [0, 1, 0, 1, 0]
            # scores = [5, 4, 3, 2, 1]
            scores = np.arange(start=len(rel_rank_mask), stop=0, step=-1)

            # rel_rank_mask is binary, so this will yield a value only for relevant positions
            # e.g.
            # rel_rank_mask = [0, 1, 0, 1, 0]
            # scores = [5, 4, 3, 2, 1]
            # rel_scores = [0, 4, 0, 2, 0]
            rel_scores = rel_rank_mask * scores

            ndcgs.append(self._calc_ndcg(rel_scores))

        return np.mean(ndcgs).item()
