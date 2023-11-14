import statistics
import unittest

import numpy as np

from src.evaluate.abstract_metric import PaddedArr, LaikaMetric
from src.evaluate.metrics.ranking_metrics import Hit, MAP, MRR, NDCG


class TestRankingMetric(unittest.TestCase):

    def test_operator_comparison(self):

        # instantiate any subclass to test method of the abstract class
        metric = Hit()

        # to find the best ranking result, we should maximize the metric
        result = metric.operator_comparison(3, 5)

        # 3 is less than 5, not greater than
        self.assertFalse(result)

    def test_per_user_computed_matrix(self):

        # for ranking metrics, with the per_user_computed_matrix we basically compute
        # the binary relevance matrix for each user

        # instantiate any subclass to test method of the abstract class
        metric = Hit()

        # ideal case (no padding needed) with only one item in the ground truth
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_2"]
        ])

        truths = PaddedArr([
            ["item_3"],
            ["item_8"],
            ["item_8"]
        ])

        result = metric.per_user_precomputed_matrix(predictions, truths)

        expected = np.array([
            [False, False, True],  # only item_3 is relevant
            [False, False, False],  # no item relevant predicted
            [True, False, False]  # only item_8 is relevant
        ])

        self.assertTrue(np.array_equal(expected, result))

        # padding needed for truth
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        result = metric.per_user_precomputed_matrix(predictions, truths)

        user_1_binary_rel = [True, False, True]  # item_1 and item_3 relevant
        user_2_binary_rel = [False, False, False]  # no relevant item
        user_3_binary_rel = [True, False, False]  # item_8 relevant

        expected = np.array([
            user_1_binary_rel,
            user_2_binary_rel,
            user_3_binary_rel
        ])

        self.assertTrue(np.array_equal(expected, result))

        # padding needed for truth and predictions cut to K
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        metric_k = Hit(k=2)
        result = metric_k.per_user_precomputed_matrix(predictions, truths)

        user_1_binary_rel = [True, False]  # item_1 relevant in pred@2
        user_2_binary_rel = [False, False]  # no relevant item in pred@2
        user_3_binary_rel = [True, False]  # item_8 relevant in pred@2

        expected = np.array([
            user_1_binary_rel,
            user_2_binary_rel,
            user_3_binary_rel
        ])

        self.assertTrue(np.array_equal(expected, result))


class TestHit(unittest.TestCase):

    def test__call__(self):

        metric = Hit()

        # ground truth only one item
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3"],
            ["item_8"],
            ["item_8"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        # item_3 relevant for user_1 -> hit
        # no item relevant for user_2 -> no hit
        # item_8 relevant for user_3 -> hit
        expected = statistics.mean([1, 0, 1])

        self.assertEqual(expected, result)

        # ground truth more than one item (padding needed)
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        # item_3 and item_1 relevant for user_1 -> hit
        # no item relevant for user_2 -> no hit
        # item_8 relevant for user_3 -> hit
        expected = statistics.mean([1, 0, 1])

        self.assertEqual(expected, result)

    def test__call__at_k(self):
        # ground truth more than one item (padding needed)

        metric = Hit(k=1)

        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_9", "item_8", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        # item_1 relevant for user_1 in pred@1 -> hit
        # no item relevant for user_2 in pred@1 -> no hit
        # no item relevant for user_3 in pred@1 -> hit
        expected = statistics.mean([1, 0, 0])

        self.assertEqual(expected, result)

    def test_from_string(self):

        # check from_string() works for this metric
        expected = Hit()
        result = LaikaMetric.from_string("hit")

        self.assertEqual(expected, result)

        # check from_string() works for this metric with case-insensitive name
        expected = Hit()
        result = LaikaMetric.from_string("hIT")

        self.assertEqual(expected, result)

        # check from_string() with k specified
        expected = Hit(k=8)
        result = LaikaMetric.from_string("hit@8")

        self.assertEqual(expected, result)

    def test_metric_exists(self):

        # check metric_exists() works for this metric
        expected = Hit
        result = LaikaMetric.metric_exists("hit", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with case-insensitive name
        expected = Hit
        result = LaikaMetric.metric_exists("hiT", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with return bool
        result = LaikaMetric.metric_exists("hit", return_bool=True)

        self.assertTrue(result)

        # check metric_exists() does not care for k even if it's not expected for the specific
        # metric
        expected = Hit
        result = LaikaMetric.metric_exists("hit@8", return_bool=False)

        self.assertEqual(expected, result)


class TestMap(unittest.TestCase):

    def test__call__(self):

        metric = MAP()

        # This is taken from recmetrics, to check if we obtain same results
        prediction = np.array([['X', 'Y', 'Z'], ['A', 'Z', 'B']])
        truth = PaddedArr([['A', 'B', 'X'], ['A', 'B', 'Y']])

        rel_binary_matrix = metric.per_user_precomputed_matrix(prediction, truth)

        ap_user_1 = 1
        ap_user_2 = 1/2 * (1 + 2/3)

        expected = (ap_user_1 + ap_user_2) / 2
        result = metric(rel_binary_matrix)

        self.assertEqual(expected, result)

        # ground truth only one item
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3"],
            ["item_8"],
            ["item_8"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        ap_user_1 = 1 / 3
        ap_user_2 = 0
        ap_user_3 = 1
        expected = (ap_user_1 + ap_user_2 + ap_user_3) / 3

        self.assertEqual(expected, result)

        # ground truth more than one item (padding needed)
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        ap_user_1 = (1 + 2/3) / 2
        ap_user_2 = 0
        ap_user_3 = 1

        expected = (ap_user_1 + ap_user_2 + ap_user_3) / 3

        self.assertEqual(expected, result)

    def test__call__at_k(self):
        # ground truth more than one item (padding needed)

        metric = MAP(k=2)

        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_9", "item_8", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        ap_user_1 = 1
        ap_user_2 = 0
        ap_user_3 = 1 / 2

        expected = (ap_user_1 + ap_user_2 + ap_user_3) / 3

        self.assertEqual(expected, result)

    def test_from_string(self):

        # check from_string() works for this metric
        expected = MAP()
        result = LaikaMetric.from_string("map")

        self.assertEqual(expected, result)

        # check from_string() works for this metric with case-insensitive name
        expected = MAP()
        result = LaikaMetric.from_string("mAp")

        self.assertEqual(expected, result)

        # check from_string() with k specified
        expected = MAP(k=8)
        result = LaikaMetric.from_string("map@8")

        self.assertEqual(expected, result)

    def test_metric_exists(self):

        # check metric_exists() works for this metric
        expected = MAP
        result = LaikaMetric.metric_exists("map", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with case-insensitive name
        expected = MAP
        result = LaikaMetric.metric_exists("maP", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with return bool
        result = LaikaMetric.metric_exists("map", return_bool=True)

        self.assertTrue(result)

        # check metric_exists() does not care for k even if it's not expected for the specific
        # metric
        expected = MAP
        result = LaikaMetric.metric_exists("map@8", return_bool=False)

        self.assertEqual(expected, result)


class TestMRR(unittest.TestCase):

    def test__call__(self):

        metric = MRR()

        # ground truth only one item
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3"],
            ["item_8"],
            ["item_8"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        rr_user_1 = 1 / 3  # first item rel for user 1 is at position 3
        rr_user_2 = 0  # no item rel predicted for user 2
        rr_user_3 = 1  # first item rel for user 3 is at position 1
        expected = (rr_user_1 + rr_user_2 + rr_user_3) / 3

        self.assertEqual(expected, result)

        # ground truth more than one item (padding needed)
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        rr_user_1 = 1  # first item rel for user 1 is at position 1
        rr_user_2 = 0  # no item rel predicted for user 2
        rr_user_3 = 1  # first item rel for user 3 is at position 1

        expected = (rr_user_1 + rr_user_2 + rr_user_3) / 3

        self.assertEqual(expected, result)

    def test__call__at_k(self):
        # ground truth more than one item (padding needed)

        metric = MRR(k=2)

        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_9", "item_10", "item_8"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        rr_user_1 = 1
        rr_user_2 = 0
        rr_user_3 = 0

        expected = (rr_user_1 + rr_user_2 + rr_user_3) / 3

        self.assertEqual(expected, result)
    
    def test_from_string(self):

        # check from_string() works for this metric
        expected = MRR()
        result = LaikaMetric.from_string("mrr")

        self.assertEqual(expected, result)

        # check from_string() works for this metric with case-insensitive name
        expected = MRR()
        result = LaikaMetric.from_string("mRr")

        self.assertEqual(expected, result)

        # check from_string() with k specified
        expected = MRR(k=8)
        result = LaikaMetric.from_string("mrr@8")

        self.assertEqual(expected, result)

    def test_metric_exists(self):
        # check metric_exists() works for this metric
        expected = MRR
        result = LaikaMetric.metric_exists("mrr", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with case-insensitive name
        expected = MRR
        result = LaikaMetric.metric_exists("mrR", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with return bool
        result = LaikaMetric.metric_exists("mrr", return_bool=True)

        self.assertTrue(result)

        # check metric_exists() does not care for k even if it's not expected for the specific
        # metric
        expected = MRR
        result = LaikaMetric.metric_exists("mrr@8", return_bool=False)

        self.assertEqual(expected, result)


class TestNDCG(unittest.TestCase):

    def test__call__(self):
        metric = NDCG()

        # ground truth only one item
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3"],
            ["item_8"],
            ["item_8"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        # no item_relevant were predicted for user_2, so ndcg is surely 0
        ndcg_user_2 = 0

        dcg_user_1 = sum([0, 0, 1 / (np.log2(1 + 3))])
        dcg_user_3 = sum([1 / (np.log2(1 + 1)), 0, 0])

        idcg_user_1 = sum([1 / (np.log2(1 + 1)), 0, 0])
        idcg_user_3 = sum([1 / (np.log2(1 + 1)), 0, 0])

        ndcg_user_1 = dcg_user_1 / idcg_user_1
        ndcg_user_3 = dcg_user_3 / idcg_user_3

        expected = (ndcg_user_1 + ndcg_user_2 + ndcg_user_3) / 3

        self.assertEqual(expected, result)

        # ground truth more than one item (padding needed)
        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_8", "item_9", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        # no item_relevant were predicted for user_2, so ndcg is surely 0
        ndcg_user_2 = 0

        dcg_user_1 = sum([1 / (np.log2(1 + 1)), 0, 1 / (np.log2(1 + 3))])
        dcg_user_3 = sum([1 / (np.log2(1 + 1)), 0, 0])

        idcg_user_1 = sum([1 / (np.log2(1 + 1)), 1 / (np.log2(1 + 2)), 0])
        idcg_user_3 = sum([1 / (np.log2(1 + 1)), 0, 0])

        ndcg_user_1 = dcg_user_1 / idcg_user_1
        ndcg_user_3 = dcg_user_3 / idcg_user_3

        expected = (ndcg_user_1 + ndcg_user_2 + ndcg_user_3) / 3

        self.assertEqual(expected, result)
    
    def test__call__at_k(self):
        # ground truth more than one item (padding needed)

        metric = NDCG(k=1)

        predictions = np.array([
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_50", "item_6"],
            ["item_9", "item_8", "item_10"]
        ])

        truths = PaddedArr([
            ["item_3", "item_1", "item_80", "item_90"],
            ["item_8", "item_3", "item_4"],
            ["item_8", "item_2"]
        ])

        rel_binary_matrix = metric.per_user_precomputed_matrix(predictions, truths)
        result = metric(rel_binary_matrix)

        # no item_relevant were predicted for user_2 and user_3 in pred@1, so ndcg is surely 0
        ndcg_user_2 = 0
        ndcg_user_3 = 0

        dcg_user_1 = sum([1 / (np.log2(1 + 1)), 0, 0])
        idcg_user_1 = sum([1 / (np.log2(1 + 1)), 0, 0])

        ndcg_user_1 = dcg_user_1 / idcg_user_1

        expected = (ndcg_user_1 + ndcg_user_2 + ndcg_user_3) / 3

        self.assertEqual(expected, result)

    def test_from_string(self):

        # check from_string() works for this metric
        expected = NDCG()
        result = LaikaMetric.from_string("ndcg")

        self.assertEqual(expected, result)

        # check from_string() works for this metric with case-insensitive name
        expected = NDCG()
        result = LaikaMetric.from_string("nDCg")

        self.assertEqual(expected, result)

        # check from_string() with k specified
        expected = NDCG(k=8)
        result = LaikaMetric.from_string("ndcg@8")

        self.assertEqual(expected, result)

    def test_metric_exists(self):
        # check metric_exists() works for this metric
        expected = NDCG
        result = LaikaMetric.metric_exists("ndcg", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with case-insensitive name
        expected = NDCG
        result = LaikaMetric.metric_exists("NDcG", return_bool=False)

        self.assertEqual(expected, result)

        # check metric_exists() works for this metric with return bool
        result = LaikaMetric.metric_exists("ndcg", return_bool=True)

        self.assertTrue(result)

        # check metric_exists() does not care for k even if it's not expected for the specific
        # metric
        expected = NDCG
        result = LaikaMetric.metric_exists("ndcg@8", return_bool=False)

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
