import os
from collections import defaultdict

import pandas as pd
from datasets import Dataset

from src import SharedParams, MODELS_DIR, METRICS_DIR
from src.data.abstract_dataset import LaikaDataset
from src.evaluate import EvalParams
from src.evaluate.evaluator import RecEvaluator
from src.model import LaikaModel
from src.utils import log_wandb


def eval_main(shared_params: SharedParams, eval_params: EvalParams, dataset_obj: LaikaDataset,
              rec_model: LaikaModel):

    device = shared_params.device
    should_log = shared_params.log_wandb

    eval_batch_size = eval_params.eval_batch_size
    metric_list_str = eval_params.metrics

    ds_dict = dataset_obj.get_hf_datasets()
    test_set = ds_dict["test"]

    # REDUCE FOR TEST
    test_set = Dataset.from_dict(test_set[:100])

    # set model to correct device
    rec_model.to(device)

    # eval

    # TO DO: it should be possible evaluate on custom tasks and not necessarily those
    # observed during training
    task_to_evaluate = rec_model.training_tasks

    evaluator = RecEvaluator(rec_model, eval_batch_size)

    for task in task_to_evaluate:

        # metrics names are keys, values are lists containing results for each template
        task_result = defaultdict(list)

        template_ids_to_evaluate = task.valid_templates(return_id=True)
        for template_id in template_ids_to_evaluate:

            print(f"Evaluating on {task}/{template_id}")

            # we don't call set_eval_task because task are already instantiated
            rec_model.eval_task = task.force_template(template_id)
            res_dict = evaluator.evaluate(test_set, metric_list_str=metric_list_str)

            dict_to_log = {f"test/{task}/template_id": template_id}
            for metric_name, metric_val in res_dict.items():
                dict_to_log[f"test/{task}/{metric_name}"] = metric_val
                task_result[metric_name].append(metric_val)

            log_wandb(dict_to_log, should_log)

        task_result_df = pd.DataFrame(task_result,
                                      index=pd.Index(template_ids_to_evaluate, name="Template id"))

        task_result_df_mean_max = task_result_df.agg({metric_name: ["mean", "max"]
                                                      for metric_name in task_result})

        log_wandb({f"test/{task}/{metric}/mean": task_result_df_mean_max[metric]["mean"]
                   for metric in task_result}, should_log)

        log_wandb({f"test/{task}/{metric}/max": task_result_df_mean_max[metric]["max"]
                   for metric in task_result}, should_log)

        print(f"Mean and max result for task {task}:")
        print(task_result_df_mean_max)

        # locally we save a single df for each task containing result for each template ids + mean and max
        task_result_df = pd.concat((task_result_df, task_result_df_mean_max))

        output_path = os.path.join(METRICS_DIR, shared_params.exp_name)
        os.makedirs(output_path, exist_ok=True)

        # e.g. SequentialSideInfo.csv
        task_result_df.to_csv(str(task))
