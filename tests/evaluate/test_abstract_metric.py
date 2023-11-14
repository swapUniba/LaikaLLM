import inspect
import unittest

import numpy as np

from src.evaluate.abstract_metric import PaddedArr, LaikaMetric, Loss


class TestPaddedArr(unittest.TestCase):

    def test__new__empty(self):

        result = PaddedArr([[]])

        # collection of empty collection, so axis 0 is 1 and axis 1 is 0
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 0)

    def test__new__no_pad(self):

        # single row, there's nothing to pad
        list_rows = [
            ["1", "2", "3", "4", "5"]
        ]

        expected = np.array(list_rows)
        result = PaddedArr(list_rows)

        # since there's nothing to pad, we expect the same result
        # we would obtain by calling directly np.array(...)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.issubdtype(result.dtype, str))
        self.assertTrue(np.array_equal(expected, result))

        # two row same dimension, there's nothing to pad
        list_rows = [
            ["1", "2", "3", "4", "5"],
            ["1", "2", "3", "4", "5"]
        ]

        expected = np.array(list_rows)
        result = PaddedArr(list_rows)

        # since there's nothing to pad, we expect the same result
        # we would obtain by calling directly np.array(...)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.issubdtype(result.dtype, str))
        self.assertTrue(np.array_equal(expected, result))

    def test__new__w_pad(self):

        # different dimensions, pad happens
        list_rows = [
            ["1", "2", "3", "4", "5"],
            ["1", "2", "3"],
            ["1"]
        ]

        expected = np.array([
            ["1", "2", "3", "4", "5"],
            ["1", "2", "3", "<PAD>", "<PAD>"],
            ["1", "<PAD>", "<PAD>", "<PAD>", "<PAD>"]
        ])
        result = PaddedArr(list_rows)

        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.issubdtype(result.dtype, str))
        self.assertTrue(np.array_equal(expected, result))

    def test__new__error(self):

        # single row, no padding needed but <PAD> token inside the collection
        list_rows = [["1", "2", "<PAD>"]]
        with self.assertRaises(AssertionError):
            PaddedArr(list_rows)

        # different dimensions, pad should happen but <PAD> token inside the collection
        list_rows = [
            ["1", "2", "<PAD>", "4", "5"],
            ["1", "2", "3"],
            ["1"]
        ]

        with self.assertRaises(AssertionError):
            PaddedArr(list_rows)


class TestLaikaMetric(unittest.TestCase):

    def test_all_metrics_available(self):

        all_cls_metrics = LaikaMetric.all_metrics_available(return_str=False)
        self.assertTrue(len(all_cls_metrics) != 0)

        for cls_metric in all_cls_metrics:

            # check that instances of LaikaMetric are returned
            self.assertTrue(issubclass(cls_metric, LaikaMetric))

            # we expect only metrics that are instantiable
            self.assertFalse(inspect.isabstract(cls_metric))

        # the string representation of each metric is the name of the class
        expected = [cls_metric.__name__ for cls_metric in all_cls_metrics]
        all_str_metrics = LaikaMetric.all_metrics_available(return_str=True)

        self.assertTrue(len(all_str_metrics) != 0)
        self.assertEqual(expected, all_str_metrics)

    def test_from_string(self):

        # this method is tested by each individual metric, at the abstract level
        # we check only that non-existent metric raise error

        # test non-existing metric
        with self.assertRaises(KeyError):
            LaikaMetric.from_string("non-existent")

    def test_metric_exists(self):

        # this method is tested by each individual metric, at the abstract level
        # we check only that non-existent metric raise error

        # test non-existing metric
        with self.assertRaises(KeyError):
            LaikaMetric.metric_exists("non-existent")
            LaikaMetric.metric_exists("non-existent", return_bool=True)

    def test_safe_div(self):

        # no 0 in denominator, so result would be the same of performing directly the division
        numerator = np.array([1, 2, 3, 4, 5])
        denominator = np.array([1, 4, 6, 16, 15])

        expected = numerator / denominator
        result = LaikaMetric.safe_div(numerator, denominator)

        self.assertTrue(np.array_equal(expected, result))

        # 0 in denominator, result is 0 wherever the division would give us nan
        numerator = np.array([1, 2, 3, 4, 5])
        denominator = np.array([2, 4, 0, 10, 0])

        expected = np.array([.5, .5, 0, .4, 0])
        result = LaikaMetric.safe_div(numerator, denominator)

        self.assertTrue(np.array_equal(expected, result))


class TestLoss(unittest.TestCase):

    def test_operator_comparison(self):

        metric = Loss()

        # to find the best loss, we must minimize it
        res = metric.operator_comparison(3, 5)

        self.assertTrue(res)

    def test_per_user_precomputed_matrix(self):

        metric = Loss()

        # loss is computed by the model, and it is defined to exploit polymorphism,
        # so methods used to compute the metrics in this case raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            metric.per_user_precomputed_matrix(np.array([[]]), PaddedArr([[]]))

    def test__call__(self):
        metric = Loss()

        # loss is computed by the model, and it is defined to exploit polymorphism,
        # so methods used to compute the metrics in this case raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            metric(np.array([]), np.array([]))

    def test_from_string(self):

        # check from_string() works for this metric
        expected = Loss()
        result = LaikaMetric.from_string("loss")

        self.assertEqual(expected, result)

        # check from_string() works for this metric with case-insensitive name
        expected = Loss()
        result = LaikaMetric.from_string("LoSs")

        self.assertEqual(expected, result)

        # check from_string() with k specified returns error
        with self.assertRaises(TypeError):
            LaikaMetric.from_string("loss@8")

    def test_metric_exists(self):

        # check metric_exists() works for this metric
        expected = Loss
        result = LaikaMetric.metric_exists("loss", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with case-insensitive name
        expected = Loss
        result = LaikaMetric.metric_exists("lOSs", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with return bool
        result = LaikaMetric.metric_exists("loss", return_bool=True)

        self.assertTrue(result)

        # check metric_exists() does not care for k even if it's not expected for the specific
        # metric
        expected = Loss
        result = LaikaMetric.metric_exists("loss@invalid", return_bool=False)

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
