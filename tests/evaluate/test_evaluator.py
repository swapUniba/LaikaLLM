import os.path
import shutil
import unittest
from unittest.mock import Mock

import datasets
import numpy as np
import pandas as pd
import torch

from src.data.tasks.tasks import SequentialSideInfoTask, RatingPredictionTask
from src.evaluate.abstract_metric import Loss
from src.evaluate.evaluator import RecEvaluator
from src.evaluate.metrics.error_metrics import MAE, RMSE
from src.evaluate.metrics.ranking_metrics import Hit, MRR, MAP
from src.model import LaikaModel


class TestEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        mocked_model = Mock(spec=LaikaModel)

        mocked_model.generate_step.return_value = (
            np.array([["1", "2", "3"], ["2", "3", "4"], ["3", "4", "5"]]),
            np.array([["1", "2", "3"], ["2", "3", "4"], ["3", "4", "5"]]),
            torch.tensor(.5)
        )

        mocked_dataset = Mock(spec=datasets.Dataset, num_rows=5)
        mocked_dataset.map.return_value = mocked_dataset
        mocked_dataset.set_format.return_value = mocked_dataset
        mocked_dataset.iter.return_value = range(5)

        cls.mocked_model = mocked_model
        cls.mocked_dataset = mocked_dataset

    def test_evaluate_suite(self):
        eva = RecEvaluator(self.mocked_model, eval_batch_size=1)

        # no latex table
        all_res = eva.evaluate_suite(
            eval_dataset=self.mocked_dataset,
            tasks_to_evaluate={
                SequentialSideInfoTask(): [Hit(k=10), Loss()],
                RatingPredictionTask(): [RMSE(), MAE()]
            },
            output_dir="to_del",
            create_latex_table=False
        )

        # all res is a dict, keys are str of tasks, values are dataframe with results
        # for each template
        self.assertEqual(len(all_res), 2)
        self.assertIn("SequentialSideInfoTask", all_res)
        self.assertIn("RatingPredictionTask", all_res)

        sequential_sideinfo_res = all_res["SequentialSideInfoTask"]
        rating_prediction_res = all_res["RatingPredictionTask"]

        self.assertIsInstance(sequential_sideinfo_res, pd.DataFrame)
        self.assertIsInstance(rating_prediction_res, pd.DataFrame)

        # assert that mean and max are present in each result df at the last 2 positions
        self.assertEqual(sequential_sideinfo_res.index[-2], "mean")
        self.assertEqual(sequential_sideinfo_res.index[-1], "max")
        self.assertEqual(rating_prediction_res.index[-2], "mean")
        self.assertEqual(rating_prediction_res.index[-1], "max")

        # assert that other index values are all the inference templates of the task
        self.assertEqual(sequential_sideinfo_res.index[:-2].tolist(),
                         SequentialSideInfoTask().inference_templates(return_id=True))
        self.assertEqual(rating_prediction_res.index[:-2].tolist(),
                         RatingPredictionTask().inference_templates(return_id=True))

        # check that csv of results are saved
        self.assertEqual(len(os.listdir("to_del")), 2)
        self.assertTrue(os.path.isfile("to_del/SequentialSideInfoTask.csv"))
        self.assertTrue(os.path.isfile("to_del/RatingPredictionTask.csv"))

        shutil.rmtree("to_del", ignore_errors=True)

        # with latex table. We don't test the result dict as it would
        # be the same of above
        eva.evaluate_suite(
            eval_dataset=self.mocked_dataset,
            tasks_to_evaluate={
                SequentialSideInfoTask(): [Hit(k=10), Loss()],
                RatingPredictionTask(): [RMSE(), MAE()]
            },
            output_dir="to_del",
            create_latex_table=True
        )

        self.assertEqual(len(os.listdir("to_del")), 4)
        self.assertTrue(os.path.isfile("to_del/SequentialSideInfoTask.csv"))
        self.assertTrue(os.path.isfile("to_del/SequentialSideInfoTask_latex.tex"))
        self.assertTrue(os.path.isfile("to_del/RatingPredictionTask.csv"))
        self.assertTrue(os.path.isfile("to_del/RatingPredictionTask_latex.tex"))

        shutil.rmtree("to_del", ignore_errors=True)

    def test_evaluate_task(self):
        eva = RecEvaluator(self.mocked_model, eval_batch_size=1)

        # SequentialSideInfo is not compatible with error metrics, exception is raised
        with self.assertRaises(ValueError):
            eva.evaluate_task(eval_dataset=self.mocked_dataset,
                              metric_list=[MAE()],
                              task=SequentialSideInfoTask(),
                              template_id=0)

        # no loss
        res = eva.evaluate_task(eval_dataset=self.mocked_dataset,
                                metric_list=[Hit(), MRR(k=2)],
                                task=SequentialSideInfoTask(),
                                template_id=0)

        # we could assert if the mocked function were called, but at this level
        # we are only interested that results that we expect are computed

        self.assertTrue(len(res) == 2)
        self.assertIn("Hit", res)
        self.assertIn("MRR@2", res)

        # the mocked 'generate_step()' returns predictions equals to truth, so
        # we expect perfect score
        self.assertEqual(res["Hit"], 1)
        self.assertEqual(res["MRR@2"], 1)

        # metrics + loss
        res = eva.evaluate_task(eval_dataset=self.mocked_dataset,
                                metric_list=[Hit(), Loss()],
                                task=SequentialSideInfoTask(),
                                template_id=0)

        # we could assert if the mocked function were called, but at this level
        # we are only interested that results that we expect are computed

        self.assertTrue(len(res) == 2)
        self.assertIn("Hit", res)
        self.assertIn("Loss", res)

        # the mocked 'generate_step()' returns predictions equals to truth, so
        # we expect perfect score
        self.assertEqual(res["Hit"], 1)
        self.assertEqual(res["Loss"], .5)  # loss is always .5 in the mocked 'generate_step()'

        # only loss
        res = eva.evaluate_task(eval_dataset=self.mocked_dataset,
                                metric_list=[Loss()],
                                task=SequentialSideInfoTask(),
                                template_id=0)

        # we could assert if the mocked function were called, but at this level
        # we are only interested that results that we expect are computed

        self.assertTrue(len(res) == 1)
        self.assertIn("Loss", res)

        # the mocked 'generate_step()' returns always the same loss,
        # so after we divide the summed loss over the total number of batches
        # we should get same loss
        self.assertEqual(res["Loss"], .5)

    # usually private methods are automatically tested when tested other methods,
    # but due to the great importance and relevance of this method, it is tested
    # individually
    def test__compute_metrics(self):

        predictions = [
            np.array(["item_1", "item_2", "item_3"]),
            np.array(["item_1", "item_50", "item_6"]),
            np.array(["item_9", "item_8", "item_2"])
        ]

        truths = [
            np.array(["item_3", "item_1", "item_80", "item_90"]),
            np.array(["item_8", "item_3", "item_4"]),
            np.array(["item_8", "item_2"])
        ]

        eva = RecEvaluator(self.mocked_model, eval_batch_size=1)

        # evaluate on metrics with no k set
        res = eva._compute_metrics(predictions, truths, metric_list=[Hit(), MRR(), MAP()])

        self.assertIn("Hit", res)
        self.assertIn("MRR", res)
        self.assertIn("MAP", res)

        hit_user_1 = 1
        hit_user_2 = 0
        hit_user_3 = 1
        expected_hit = (hit_user_1 + hit_user_2 + hit_user_3) / 3

        rr_user_1 = 1
        rr_user_2 = 0
        rr_user_3 = 1/2
        expected_mrr = (rr_user_1 + rr_user_2 + rr_user_3) / 3

        ap_user_1 = (1 + 2/3) / 2
        ap_user_2 = 0
        ap_user_3 = (1/2 + 2/3) / 2
        expected_map = (ap_user_1 + ap_user_2 + ap_user_3) / 3

        self.assertEqual(res["Hit"], expected_hit)
        self.assertEqual(res["MRR"], expected_mrr)
        self.assertEqual(res["MAP"], expected_map)

        # evaluate on metrics with different k set
        res = eva._compute_metrics(predictions, truths, metric_list=[Hit(k=1), MRR(k=2), MAP(k=None)])

        self.assertIn("Hit@1", res)
        self.assertIn("MRR@2", res)
        self.assertIn("MAP", res)

        hit_user_1 = 1
        hit_user_2 = 0
        hit_user_3 = 0
        expected_hit = (hit_user_1 + hit_user_2 + hit_user_3) / 3

        rr_user_1 = 1
        rr_user_2 = 0
        rr_user_3 = 1/2
        expected_mrr = (rr_user_1 + rr_user_2 + rr_user_3) / 3

        ap_user_1 = (1 + 2/3) / 2
        ap_user_2 = 0
        ap_user_3 = (1/2 + 2/3) / 2
        expected_map = (ap_user_1 + ap_user_2 + ap_user_3) / 3

        self.assertEqual(res["Hit@1"], expected_hit)
        self.assertEqual(res["MRR@2"], expected_mrr)
        self.assertEqual(res["MAP"], expected_map)

        # evaluate on metrics with same k set
        res = eva._compute_metrics(predictions, truths, metric_list=[Hit(k=1), MRR(k=1), MAP(k=1)])

        self.assertIn("Hit@1", res)
        self.assertIn("MRR@1", res)
        self.assertIn("MAP@1", res)

        hit_user_1 = 1
        hit_user_2 = 0
        hit_user_3 = 0
        expected_hit = (hit_user_1 + hit_user_2 + hit_user_3) / 3

        rr_user_1 = 1
        rr_user_2 = 0
        rr_user_3 = 0
        expected_mrr = (rr_user_1 + rr_user_2 + rr_user_3) / 3

        ap_user_1 = 1
        ap_user_2 = 0
        ap_user_3 = 0
        expected_map = (ap_user_1 + ap_user_2 + ap_user_3) / 3

        self.assertEqual(res["Hit@1"], expected_hit)
        self.assertEqual(res["MRR@1"], expected_mrr)
        self.assertEqual(res["MAP@1"], expected_map)

    def test__compute_metrics_raise_error(self):

        # predictions can't have <PAD> as prediction,
        # it is the reserved token for padding
        predictions = [
            np.array(["<PAD>", "item_1", "item_2"])
        ]

        truths = [
            np.array(["gt_1", "gt_2", "gt_3"])
        ]

        eva = RecEvaluator(self.mocked_model, eval_batch_size=1)

        with self.assertRaises(AssertionError):
            eva._compute_metrics(predictions, truths, metric_list=[Hit()])


if __name__ == '__main__':
    unittest.main()
