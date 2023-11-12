import os

from datasets import Dataset

from src import GeneralParams, METRICS_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.data import DataParams
from src.data.abstract_dataset import LaikaDataset
from src.data.abstract_task import LaikaTask
from src.evaluate import EvalParams
from src.evaluate.abstract_metric import LaikaMetric
from src.evaluate.evaluator import RecEvaluator
from src.model import LaikaModel, ModelParams


def eval_main(general_params: GeneralParams, data_params: DataParams, model_params: ModelParams, eval_params: EvalParams):

    # general params
    exp_name = general_params.exp_name
    device = general_params.device
    should_log = general_params.log_wandb

    # eval params
    eval_batch_size = eval_params.eval_batch_size
    eval_task_dict = eval_params.eval_tasks
    create_latex_table = eval_params.create_latex_table

    # load dataset created in data phase
    dataset_cls = LaikaDataset.dataset_exists(data_params.dataset_cls_name, return_bool=False)

    dataset_path = os.path.join(PROCESSED_DATA_DIR, exp_name)
    dataset_obj = dataset_cls.load(dataset_path)

    # load model created in model phase
    model_cls = LaikaModel.model_exists(model_params.model_cls_name, return_bool=False)

    model_path = os.path.join(MODELS_DIR, general_params.exp_name)
    rec_model = model_cls.load(model_path, **model_params.model_kwargs)

    ds_dict = dataset_obj.get_hf_datasets()
    test_set = ds_dict["test"]

    # REDUCE FOR TEST
    # test_set = Dataset.from_dict(test_set[:100])

    # set model to correct device
    rec_model.to(device)

    # convert from str to objects
    eval_task_dict = {
        LaikaTask.from_string(eval_task_str): [LaikaMetric.from_string(metric_str) for metric_str in metric_list_str]
        for eval_task_str, metric_list_str in eval_task_dict.items()
    }

    output_dir = os.path.join(METRICS_DIR, exp_name)

    evaluator = RecEvaluator(rec_model, eval_batch_size, should_log=should_log)

    evaluator.evaluate_suite(test_set,
                             tasks_to_evaluate=eval_task_dict,
                             output_dir=output_dir,
                             create_latex_table=create_latex_table)
