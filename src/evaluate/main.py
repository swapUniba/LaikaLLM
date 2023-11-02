import os

from datasets import Dataset

from src import SharedParams, METRICS_DIR
from src.data.abstract_dataset import LaikaDataset
from src.data.abstract_templates import Task
from src.evaluate import EvalParams
from src.evaluate.abstract_metric import Metric
from src.evaluate.evaluator import RecEvaluator
from src.model import LaikaModel


def eval_main(shared_params: SharedParams, eval_params: EvalParams, dataset_obj: LaikaDataset,
              rec_model: LaikaModel):
    exp_name = shared_params.exp_name
    device = shared_params.device
    should_log = shared_params.log_wandb

    eval_batch_size = eval_params.eval_batch_size
    metric_list_str = eval_params.metrics
    eval_tasks_str = eval_params.eval_tasks
    create_latex_table = eval_params.create_latex_table

    ds_dict = dataset_obj.get_hf_datasets()
    test_set = ds_dict["test"]

    # REDUCE FOR TEST
    test_set = Dataset.from_dict(test_set[:100])

    # set model to correct device
    rec_model.to(device)

    # convert from str to objects
    tasks_to_evaluate = Task.from_string(*eval_tasks_str, all_unique_items=dataset_obj.all_items)
    metric_list = Metric.from_string(*metric_list_str)

    output_dir = os.path.join(METRICS_DIR, exp_name)

    evaluator = RecEvaluator(rec_model, eval_batch_size, should_log=should_log)

    evaluator.evaluate_suite(test_set,
                             tasks_to_evaluate=tasks_to_evaluate,
                             metric_list=metric_list,
                             output_dir=output_dir,
                             create_latex_table=create_latex_table)
