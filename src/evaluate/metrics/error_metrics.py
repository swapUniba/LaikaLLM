import operator

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluate.abstract_metric import LaikaMetric, PaddedArr


class ErrorMetric(LaikaMetric):

    def __init__(self):
        super().__init__(k=None)

    @property
    def operator_comparison(self):
        return operator.lt

    def per_user_precomputed_matrix(self, predictions: np.ndarray[str], truths: PaddedArr):

        if (predictions.shape != truths.shape) or (truths == "<PAD>").any():
            raise ValueError("When computing Error metrics, predictions and truths should be in 1:1 relationship and "
                             "thus have same shape!")

        predictions = pd.to_numeric(predictions.flatten(), errors="coerce")
        truths = pd.to_numeric(truths.flatten(), errors="coerce")

        if np.isnan(truths).any():
            raise ValueError("Array representing the ground truth contains elements which are not numbers, "
                             "but numbers are required for error metrics!")

        # we consider only valid predictions (and their corresponding truth values),
        # i.e. generated text by the LLM which can be converted into a number
        nan_predictions = np.isnan(predictions)

        valid_preds = predictions[~nan_predictions]
        valid_truths = truths[~nan_predictions]

        if len(valid_preds) != len(predictions):
            ignored_users = len(predictions) - len(valid_preds)
            logger.info(f"For metric {str(self)}, {ignored_users} users are ignored since the LLM did not "
                        f"generate valid numbers")

        # we are bounding the predictions made which are over/below
        # the range of values we have in truth
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
