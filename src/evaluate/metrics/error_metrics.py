import operator

import numpy as np
import pandas as pd

from src.evaluate.abstract_metric import LaikaMetric, PaddedArr


class ErrorMetric(LaikaMetric):

    def __init__(self):
        super().__init__(k=None)

    @property
    def operator_comparison(self):
        return operator.lt

    def per_user_precomputed_matrix(self, predictions: np.ndarray[np.ndarray[str]], truths: PaddedArr):

        if (predictions.shape != truths.shape) or (truths == "<PAD>").any():
            raise ValueError("When computing Error metrics, predictions and truths should be in 1:1 relationship and "
                             "thus have same shape!")

        predictions = pd.to_numeric(predictions.flatten(), errors="coerce")
        truths = pd.to_numeric(truths.flatten(), errors="coerce")

        valid_preds = predictions[~np.isnan(predictions)]
        valid_truths = truths[~np.isnan(truths)]

        max_truth = valid_truths.max()
        min_truth = valid_truths.min()

        valid_preds[valid_preds > max_truth] = max_truth
        valid_preds[valid_preds < min_truth] = min_truth

        # perform subtraction ignoring <PAD> tokens
        return valid_preds - valid_truths


class RMSE(ErrorMetric):

    def __call__(self, per_user_precomputed_matrix: np.ndarray) -> float:

        return np.sqrt(np.mean(per_user_precomputed_matrix ** 2)).item()


class MAE(ErrorMetric):

    def __call__(self, per_user_precomputed_matrix: np.ndarray) -> float:

        return np.mean(np.abs(per_user_precomputed_matrix)).item()
