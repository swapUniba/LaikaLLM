import operator
import os
from collections import defaultdict
from math import ceil

import datasets
import pandas as pd
import wandb
from tqdm import tqdm

from src.data.abstract_task import Task
from src.evaluate.abstract_metric import LaikaMetric, PaddedArr
from src.evaluate.abstract_metric import Loss
from src.model import LaikaModel
from src.utils import log_wandb


class RecEvaluator:

    def __init__(self, rec_model: LaikaModel, eval_batch_size: int, should_log: bool = False):
        self.rec_model = rec_model
        self.eval_batch_size = eval_batch_size
        self.should_log = should_log

    def evaluate_suite(self,
                       eval_dataset: datasets.Dataset,
                       tasks_to_evaluate: dict[Task, list[LaikaMetric]],
                       output_dir: str,
                       create_latex_table: bool = True):

        print(f"# Starting evaluation on {', '.join(str(task) for task in tasks_to_evaluate)}\n")

        split_name = eval_dataset.split if eval_dataset.split is not None else "eval"

        # Log all eval templates used
        dataframe_dict = {"task_type": [], "template_id": [],
                          "input_text_placeholder": [], "target_text_placeholder": []}
        for task in tasks_to_evaluate:

            # we evaluate only on valid templates, that's why we iterate over only those
            for template_id in task.inference_templates(return_id=True):
                input_text_placeholder, target_text_placeholder = task.templates_dict[template_id]

                dataframe_dict["task_type"].append(str(task))
                dataframe_dict["template_id"].append(template_id)
                dataframe_dict["input_text_placeholder"].append(input_text_placeholder)
                dataframe_dict["target_text_placeholder"].append(target_text_placeholder)

        log_wandb({f"{split_name}/task_templates": wandb.Table(dataframe=pd.DataFrame(dataframe_dict))},
                  self.should_log)

        for i, (task, metric_list) in enumerate(tasks_to_evaluate.items(), start=1):

            # metrics names are keys, values are lists containing results for each template
            task_result = defaultdict(list)

            template_ids_to_evaluate = task.inference_templates(return_id=True)
            for template_id in template_ids_to_evaluate:

                print(f"# Evaluating on {task} - Template {template_id}")

                res_dict = self.evaluate_task(eval_dataset, metric_list=metric_list,
                                              task=task,
                                              template_id=template_id)

                dict_to_log = {f"{split_name}/{task}/template_id": template_id}
                for metric_name, metric_val in res_dict.items():
                    dict_to_log[f"{split_name}/{task}/{metric_name}"] = metric_val
                    task_result[metric_name].append(metric_val)

                log_wandb(dict_to_log, self.should_log)

                # simple newline for better separation between template evaluations
                print()

            task_result_df = pd.DataFrame(task_result, index=template_ids_to_evaluate)

            task_result_df_mean_max = task_result_df.agg({metric_name: ["mean", "max"]
                                                          for metric_name in task_result})

            log_wandb({f"{split_name}/{task}/{metric} - Mean": task_result_df_mean_max[metric]["mean"]
                       for metric in task_result}, self.should_log)

            log_wandb({f"{split_name}/{task}/{metric} - Max": task_result_df_mean_max[metric]["max"]
                       for metric in task_result}, self.should_log)

            print(f"Mean and max result for task {task}:")
            print(task_result_df_mean_max)
            print()

            # locally we save a single df for each task containing result for each template ids + mean and max
            task_result_df = pd.concat((task_result_df, task_result_df_mean_max))
            task_result_df.index.name = "Template ID"

            # e.g. reports/metrics/eval_exp/SequentialSideInfo.csv
            os.makedirs(output_dir, exist_ok=True)
            task_result_df.to_csv(os.path.join(output_dir, f"{task}.csv"))

            print(f"# CSV Results saved into {os.path.join(output_dir, f'{task}.csv')}!")

            if create_latex_table is True:
                latex_table = self._create_latex_table(task_result_df, task_name=str(task))

                with open(os.path.join(output_dir, f"{task}_latex.tex"), "w") as f:
                    f.write(latex_table)

                print(f"# Latex Results saved into {os.path.join(output_dir, f'{task}_latex.tex')}!")

            if i != len(tasks_to_evaluate):
                # at the end of the whole eval process we don't print separator
                print("-" * 80)

    def evaluate_task(self, eval_dataset: datasets.Dataset,
                      metric_list: list[LaikaMetric],
                      task: Task,
                      template_id: int = None):

        all_cls_metrics = {metric.__class__ for metric in metric_list}

        for cls_metric in all_cls_metrics:

            # if metric is compatible or is Loss, we pass directly to the next metric to check
            if any(issubclass(cls_metric, cls_compatible) or cls_metric == Loss
                   for cls_compatible in task.compatible_metrics()):
                continue

            raise ValueError(
                f"Task {task} is incompatible with {cls_metric.__name__}! It can be only evaluated on the "
                f"following metrics: {[compatible_metric.__name__ for compatible_metric in task.compatible_metrics()]}"
            )

        self.rec_model.eval()

        if template_id is not None:
            task = task.force_template(template_id)

        # we don't call set_eval_task because task are already instantiated
        self.rec_model.eval_task = task

        # Loss() metric it's just a placeholder needed for exploiting polymorphism
        return_loss = False
        if Loss() in metric_list:
            return_loss = True
            metric_list.remove(Loss())

        split_name = eval_dataset.split if eval_dataset.split is not None else "eval"

        preprocessed_eval = eval_dataset.map(
            self.rec_model.tokenize,
            remove_columns=eval_dataset.column_names,
            keep_in_memory=True,
            load_from_cache_file=False,
            batched=True,
            desc=f"Tokenizing {split_name} set"
        )
        preprocessed_eval.set_format("torch")

        # ceil because we don't drop the last batch
        total_n_batch = ceil(preprocessed_eval.num_rows / self.eval_batch_size)

        pbar_eval = tqdm(preprocessed_eval.iter(batch_size=self.eval_batch_size),
                         total=total_n_batch)

        eval_loss = 0
        total_preds = []
        total_truths = []

        # progress will go from 0 to 100. Init to -1 so at 0 we perform the first print
        progress = -1
        for i, batch in enumerate(pbar_eval, start=1):

            prepared_input = self.rec_model.prepare_input(batch)
            predictions, truths, loss = self.rec_model.generate_step(prepared_input, return_loss=return_loss)

            eval_loss += loss.item()

            total_preds.extend(predictions)
            total_truths.extend(truths)

            # we update the loss every 1% progress considering the total n° of batches
            # tqdm update integer percentage (1%, 2%) when float percentage is over .5 threshold (1.501 -> 2%)
            # so we print infos in the same way
            if round(100 * (i / total_n_batch)) > progress:
                result_so_far = self._compute_metrics(total_preds, total_truths, metric_list)

                pbar_desc = []

                if return_loss:
                    pbar_desc.append(f"{split_name} Loss -> {(eval_loss / i):.6f}")

                for metric_name, metric_val in result_so_far.items():
                    pbar_desc.append(f"{metric_name} -> {metric_val:.6f}")

                pbar_eval.set_description(", ".join(pbar_desc))

                progress += 1

        pbar_eval.close()

        eval_loss /= total_n_batch

        res_eval_dict = self._compute_metrics(total_preds, total_truths, metric_list)

        if return_loss is True:
            res_eval_dict[str(Loss())] = eval_loss

        return res_eval_dict

    @staticmethod
    def _compute_metrics(preds: list[str], truths: list[str], metric_list: list[LaikaMetric]):

        # this works regardless of metric type, k is always None for error metrics. This works
        # on the assumption that a metric type is the immediate parent of the specific metric,
        # e.g. class Hit(RankingMetric) -> RankingMetric is the metric type

        # Pad array if necessary
        padded_preds = PaddedArr(preds)
        padded_truths = PaddedArr(truths)

        # used to save some computational resources, we will compute binary relevance binary for
        # predictions cut to max_k (i.e. predictions[:, :max_k]). If there is at least one None,
        # sadly it means that we can't save any resource
        max_k = None
        all_ks = [metric.k for metric in metric_list]
        if None not in all_ks:
            # if no metric has k set, default is None
            max_k = max(all_ks, default=None)

        # we are separating metrics depending on their class
        type2metric_dict: dict[type[LaikaMetric], list[LaikaMetric]] = defaultdict(list)

        for metric in metric_list:
            # metric_type is the immediate parent
            [metric_type] = metric.__class__.__bases__
            type2metric_dict[metric_type].append(metric)

        # each metric type could compute the precomputed metric differently
        # If ranking metrics:
        # Build rel binary matrix by cutting predictions to the max k desired
        # Save resources by not computing relevance for predictions outside the k range,
        # which are not used by any metric passed in input
        cls_precomputed_matrix = {
            metric_type: metric_type.per_user_precomputed_matrix(padded_preds, padded_truths, k=max_k)
            for metric_type in type2metric_dict.keys()
        }

        all_metric_results = {}
        for metric in metric_list:

            [metric_type] = metric.__class__.__bases__
            precomputed_matrix = cls_precomputed_matrix[metric_type]

            # when computing the specific metric result, we consider its k value which wil surely be <= max_k
            # obviously if k is None, all predictions will be used
            if metric.k is not None:
                precomputed_matrix = precomputed_matrix[:, :metric.k]

            all_metric_results[str(metric)] = metric(precomputed_matrix)

        return all_metric_results

    @staticmethod
    def _create_latex_table(res_df: pd.DataFrame, task_name: str):

        title = task_name
        n_metrics = len(res_df.columns)

        # preliminary code for the tex file
        latex_code = r"\documentclass{article}" + "\n"
        latex_code += r"\usepackage{booktabs}" + "\n"
        latex_code += r"\begin{document}" + " \n\n"

        # title start
        latex_code += r"\begin{tabular}{c|" + "c" * n_metrics + "}\n\n"
        latex_code += r"\multicolumn{" + str(n_metrics + 1) + r"}{c}{\textbf{" + title + r"}} \\" + "\n"
        latex_code += r"\noalign{\smallskip}" + "\n"
        latex_code += r"\noalign{\smallskip}" + "\n"
        # title end

        # table start
        latex_code += r"\toprule" + "\n"

        # --column headers start
        latex_code += r"\multicolumn{1}{c}{Template ID}" + "\t&\t"

        # first is |c
        latex_code += r"\multicolumn{1}{|c}{" + res_df.columns[0] + "}"

        # all the other column headers are c
        latex_code += "\t&\t" + "\t&\t".join(r"\multicolumn{1}{c}{" + metric_name + "}"
                                             for metric_name in res_df.columns[1:])
        latex_code += r"\\" + "\n"

        # --column headers end

        # --start numeric values
        latex_code += r"\midrule" + "\n"

        template_res = res_df[:-2]
        max_mean = res_df[-2:]

        # set bold for template id which gave best result for each metric
        for metric_name in template_res.columns:

            metric_obj = LaikaMetric.from_string(metric_name)

            # depending on the metric, best result is obtained by maximizing or minimizing
            if metric_obj.operator_comparison == operator.gt:
                best_metric_idx = template_res[metric_name].idxmax()
            else:
                best_metric_idx = template_res[metric_name].idxmin()

            template_res.loc[:, metric_name] = template_res[metric_name].map(lambda x: "%.4f" % x)
            template_res.at[best_metric_idx, metric_name] = \
                r"\textbf{" + template_res.loc[best_metric_idx, metric_name] + "}"

        # fill cell values row by row
        for index, row in template_res.iterrows():
            latex_code += f"{index}\t&\t" + "\t&\t".join(row) + r"\\" + "\n"

        # --start max mean results
        latex_code += r"\midrule" + "\n"

        # fill cell values row by row
        max_mean = max_mean.map(lambda x: "%.4f" % x)
        for index, row in max_mean.iterrows():
            latex_code += f"{index}\t&\t" + "\t&\t".join(row) + r"\\" + "\n"

        latex_code += r"\bottomrule" + "\n\n"

        latex_code += r"\end{tabular}" + "\n\n"

        latex_code += r"\end{document}" + "\n"

        return latex_code
