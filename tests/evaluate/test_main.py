import os
import unittest
from unittest.mock import Mock, patch, MagicMock

from src import GeneralParams, METRICS_DIR
from src.data import LaikaDataset, DataParams
from src.data.tasks.tasks import SequentialSideInfoTask, DirectSideInfoTask
from src.evaluate import EvalParams
from src.evaluate.abstract_metric import Loss
from src.evaluate.evaluator import RecEvaluator
from src.evaluate.main import eval_main
from src.evaluate.metrics.ranking_metrics import Hit, MRR
from src.model import LaikaModel, ModelParams


mocked_dataset_hf = MagicMock()

mocked_dataset_cls = MagicMock()
mocked_dataset_cls.load.return_value.get_hf_datasets.return_value = {"test": mocked_dataset_hf}


mocked_model_obj = MagicMock()

mocked_model_cls = Mock()
mocked_model_cls.load.return_value = mocked_model_obj


class TestEvalMain(unittest.TestCase):

    @patch.object(LaikaDataset, "dataset_exists", return_value=mocked_dataset_cls)
    @patch.object(LaikaModel, "model_exists", return_value=mocked_model_cls)
    @patch.object(RecEvaluator, "__init__", return_value=None)
    @patch.object(RecEvaluator, "evaluate_suite")
    def test_eval_main(self, mock_evaluate_suite, mock_rec_eval_init, mock_model_exists, mock_dataset_exists):

        general_params = GeneralParams(exp_name="exp_name")
        data_params = DataParams(dataset_cls_name="dataset_name", dataset_params={})
        model_params = ModelParams(model_cls_name="model_name", model_kwargs={}, train_tasks=("SequentialSideInfoTask",
                                                                                              "DirectSideInfoTask"))
        eval_params = EvalParams(eval_tasks={"SequentialSideInfoTask": ["loss", "hit@10"],
                                             "DirectSideInfoTask": ["hit@5", "mrr@1"]},
                                 eval_batch_size=1)

        eval_main(general_params, data_params, model_params, eval_params)

        mock_dataset_exists.assert_called_with("dataset_name", return_bool=False)
        mock_model_exists.assert_called_with("model_name", return_bool=False)
        mock_rec_eval_init.assert_called_with(mocked_model_obj, 1, should_log=False)

        mock_evaluate_suite.assert_called_with(mocked_dataset_hf,
                                               tasks_to_evaluate={SequentialSideInfoTask(): [Loss(), Hit(k=10)],
                                                                  DirectSideInfoTask(): [Hit(k=5), MRR(k=1)]},
                                               output_dir=os.path.join(METRICS_DIR, "exp_name"),
                                               create_latex_table=eval_params.create_latex_table)


if __name__ == '__main__':
    unittest.main()
