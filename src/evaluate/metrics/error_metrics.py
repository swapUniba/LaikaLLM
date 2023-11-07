import operator

import numpy as np

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

        float_preds = np.full_like(predictions, fill_value=-1, dtype=float)
        float_truths = np.full_like(predictions, fill_value=-1, dtype=float)

        n_errors = 0
        # if for a user some predictions contains values other than "castable-to-numeric" values,
        # it is not considered when performing the evaluation
        for i in range(predictions.shape[0]):

            user_preds = predictions[i]
            user_truths = truths[i]

            try:
                float_preds[i] = user_preds.astype(float)
                float_truths[i] = user_truths.astype(float)
            except ValueError:
                n_errors += 1

        if n_errors != 0:
            print(f"WARNING: when computing error metrics, {n_errors} users predictions where discarded "
                  f"since predictions contained other values than numeric values!")

        # bound predictions to max and min found in truths
        max_value_truth = np.max(float_truths)
        min_value_truth = np.min(float_truths)

        float_preds[float_preds > max_value_truth] = max_value_truth
        float_preds[float_preds < min_value_truth] = min_value_truth

        return np.abs(float_preds - float_truths)


class RMSE(ErrorMetric):

    def __call__(self, per_user_precomputed_matrix: np.ndarray) -> float:

        per_user_mse = (per_user_precomputed_matrix ** 2).mean(axis=1)
        per_user_rmse = np.sqrt(per_user_mse)

        return per_user_rmse.mean().item()


class MAE(ErrorMetric):

    def __call__(self, per_user_precomputed_matrix: np.ndarray) -> float:

        return per_user_precomputed_matrix.mean().item()


if __name__ == "__main__":

    m = RMSE()
    m2 = MAE()

    arr1 = np.array([
        [5, 4, 5.5, 1.2],
        [3, 1, 4, 4]
    ])

    arr2 = np.array([
        [5, 5, 5, 5],
        [4, 4, 4, 4]
    ])

    a = m.per_user_precomputed_matrix(arr1, arr2)

    print(m(a))
    print(m2(a))

