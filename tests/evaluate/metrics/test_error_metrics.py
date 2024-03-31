import unittest

import numpy as np

from src.evaluate.abstract_metric import PaddedArr, LaikaMetric
from src.evaluate.metrics.error_metrics import ErrorMetric, MAE, RMSE


class TestErrorMetric(unittest.TestCase):
    def test_operator_comparison(self):

        # instantiate any subclass to test methods of the abstract superclass
        metric = MAE()

        # to find the best loss, we must minimize it
        res = metric.operator_comparison(3, 5)

        self.assertTrue(res)

    def test_per_user_computed_metrics(self):

        # instantiate any subclass to test methods of the abstract superclass
        metric = MAE()

        # ideal case
        predictions = np.array([
            ["1.2", "4.4", "3.3"],
            ["4.2", "3.2", "5"]
        ])

        truths = PaddedArr([
            ["3", "3", "1"],
            ["4", "1", "5"]
        ])

        # in the ideal case, the precomputed metrix is simply the SUBTRACTION between
        # predictions and truths (converted to float and flattened first)
        # they are flattened because we are sure preds and truths are in 1:1 relationship
        # and error metrics system-wise are not computed by averaging over user,
        # rather by considering each pred individually
        result = metric.per_user_precomputed_matrix(predictions, truths)
        expected = predictions.astype(float).flatten() - truths.astype(float).flatten()

        self.assertTrue(np.array_equal(expected, result))

        # predictions have values "outside range"
        predictions = np.array([
            ["1.2", "100", "3.3"],
            ["4.2", "-20", "5"]
        ])

        truths = PaddedArr([
            ["3", "3", "1"],
            ["4", "1", "5"]
        ])

        # prediction values are bounded to max and min of values
        # encountered in truth (so 8 will be bound to 5 and 0 will be bound to 1)
        result = metric.per_user_precomputed_matrix(predictions, truths)

        predictions[0, 1] = "5"  # 8 is bounded to 5 (max value encountered in truth)
        predictions[1, 1] = "1"  # 0 is bounded to 1 (min value encountered in truth)
        expected = predictions.astype(float).flatten() - truths.astype(float).flatten()

        self.assertTrue(np.array_equal(expected, result))

        # predictions and truths are not in 1:1 relationship. They have different shape
        predictions = np.array([
            ["1.2", "4.4", "3.3", "5.0"],
            ["4.2", "3.2", "5", "1.1"]
        ])

        truths = PaddedArr([
            ["3", "1", "4"],
            ["4", "1", "5"]
        ])

        with self.assertRaises(ValueError):
            metric.per_user_precomputed_matrix(predictions, truths)

        # predictions and truths are not in 1:1 relationship. They have same shape after padding,
        # but one element of truth is a <PAD> token
        predictions = np.array([
            ["1.2", "4.4", "3.3"],
            ["4.2", "3.2", "5"]
        ])

        truths = PaddedArr([
            ["3", "1"],
            ["4", "1", "5"]
        ])

        with self.assertRaises(ValueError):
            metric.per_user_precomputed_matrix(predictions, truths)

        # predictions and truths are in 1:1 relationships but some predictions are invalid
        # (i.e. they are not numbers)
        predictions = np.array([
            ["not a number", "4.4", "3.3"],
            ["4.2", "not a number", "5"]
        ])

        truths = PaddedArr([
            ["3", "3", "1"],
            ["4", "1", "5"]
        ])

        result = metric.per_user_precomputed_matrix(predictions, truths)

        valid_predictions = predictions.flatten()

        valid_numbers_index = [1, 2, 3, 5]  # position of valid numbers in the flattened arr
        valid_predictions = valid_predictions[valid_numbers_index].astype(float)
        valid_truths = truths.astype(float).flatten()[valid_numbers_index]

        expected = valid_predictions - valid_truths

        self.assertTrue(np.array_equal(expected, result))

        # truths contain invalid numbers, which should be impossible
        predictions = np.array([
            ["1.2", "4.4", "3.3"],
            ["4.2", "3.2", "5"]
        ])

        truths = PaddedArr([
            ["3", "3", "1"],
            ["4", "1", "not a number"]
        ])

        with self.assertRaises(ValueError):
            metric.per_user_precomputed_matrix(predictions, truths)


# for actual metrics we only test the correctness of results,
# all limit cases are tested in test_per_user_precomputed_matrix()
class TestRMSE(unittest.TestCase):

    def test__call__(self):
        # ideal case
        predictions = np.array([
            ["1.2", "4.4", "3.3"],
            ["4.2", "3.2", "5"]
        ])

        truths = PaddedArr([
            ["3", "3", "1"],
            ["4", "1", "5"]
        ])

        metric = RMSE()

        precomputed_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(precomputed_matrix)

        # simply compute RMSE formula
        expected = (predictions.astype(float).flatten() - truths.astype(float).flatten())
        expected = expected ** 2
        expected = np.sqrt(np.mean(expected)).item()

        self.assertEqual(expected, result)

    def test_from_string(self):

        # check from_string() works for this metric
        expected = RMSE()
        result = LaikaMetric.from_string("rmse")

        self.assertEqual(expected, result)

        # check from_string() works for this metric with case-insensitive name
        expected = RMSE()
        result = LaikaMetric.from_string("rMsE")

        self.assertEqual(expected, result)

        # check from_string() with k specified returns error
        with self.assertRaises(TypeError):
            LaikaMetric.from_string("rmse@8")

    def test_metric_exists(self):

        # check metric_exists() works for this metric
        expected = RMSE
        result = LaikaMetric.metric_exists("rmse", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with case-insensitive name
        expected = RMSE
        result = LaikaMetric.metric_exists("rMSe", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with return bool
        result = LaikaMetric.metric_exists("rmse", return_bool=True)

        self.assertTrue(result)

        # check metric_exists() does not care for k even if it's not expected for the specific
        # metric
        expected = RMSE
        result = LaikaMetric.metric_exists("rmse@invalid", return_bool=False)

        self.assertEqual(expected, result)


# for actual metrics we only test the correctness of results,
# all limit cases are tested in test_per_user_precomputed_matrix()
class TestMAE(unittest.TestCase):

    def test__call__(self):
        # ideal case
        predictions = np.array([
            ["1.2", "4.4", "3.3"],
            ["4.2", "3.2", "5"]
        ])

        truths = PaddedArr([
            ["3", "3", "1"],
            ["4", "1", "5"]
        ])

        metric = MAE()

        precomputed_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(precomputed_matrix)

        # simply compute MAE formula
        expected = predictions.astype(float).flatten() - truths.astype(float).flatten()
        expected = np.mean(np.abs(expected)).item()

        self.assertEqual(expected, result)

    def test_from_string(self):

        # check from_string() works for this metric
        expected = MAE()
        result = LaikaMetric.from_string("mae")

        self.assertEqual(expected, result)

        # check from_string() works for this metric with case-insensitive name
        expected = MAE()
        result = LaikaMetric.from_string("MAe")

        self.assertEqual(expected, result)

        # check from_string() with k specified returns error
        with self.assertRaises(TypeError):
            LaikaMetric.from_string("rmse@8")

    def test_metric_exists(self):

        # check metric_exists() works for this metric
        expected = MAE
        result = LaikaMetric.metric_exists("mae", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with case-insensitive name
        expected = MAE
        result = LaikaMetric.metric_exists("MaE", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with return bool
        result = LaikaMetric.metric_exists("mae", return_bool=True)

        self.assertTrue(result)

        # check metric_exists() does not care for k even if it's not expected for the specific
        # metric
        expected = MAE
        result = LaikaMetric.metric_exists("mae@invalid", return_bool=False)

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
