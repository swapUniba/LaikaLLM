import operator

import numpy as np
import pandas as pd

from src.evaluate.abstract_metric import LaikaMetric


class ErrorMetric(LaikaMetric):

    def __init__(self):
        super().__init__(k=None)

    @property
    def operator_comparison(self):
        return operator.lt

    @staticmethod
    def per_user_precomputed_matrix(predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[np.ndarray[str]],
                                    **kwargs):
        # k is not used

        if predictions.shape != truths.shape:
            raise ValueError("When computing Error metrics, predictions and truths should have the same shape!")

        predictions = pd.to_numeric(pd.Series(predictions.flatten()), errors="coerce")
        truths = pd.to_numeric(pd.Series(truths.flatten()), errors="raise")

        valid_values = predictions.notna()

        # we ignore predictions which are not a valid number when performing subtraction
        return (predictions[valid_values] - truths[valid_values]).to_numpy()


class RMSE(ErrorMetric):

    def __call__(self, per_user_precomputed_matrix: np.ndarray) -> float:

        return np.sqrt(np.mean(per_user_precomputed_matrix ** 2)).item()


class MAE(ErrorMetric):

    def __call__(self, per_user_precomputed_matrix: np.ndarray) -> float:

        return np.mean(np.abs(per_user_precomputed_matrix)).item()
