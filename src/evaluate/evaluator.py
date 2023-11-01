from math import ceil
from typing import List, Optional, Iterable

import datasets
import numpy as np
from tqdm import tqdm

from src.evaluate.abstract_metric import Metric, PaddedArr
from src.model import LaikaModel


class RecEvaluator:

    def __init__(self, rec_model: LaikaModel, eval_batch_size: int):
        self.rec_model = rec_model
        self.eval_batch_size = eval_batch_size

    def evaluate(self, eval_dataset: datasets.Dataset, metric_list_str: Iterable[str], return_loss: bool = False):

        self.rec_model.eval()

        # convert from str to objects
        metric_list = Metric.from_string(*metric_list_str)

        # used to save some computational resources, we will compute binary relevance binary for
        # predictions cut to max_k (i.e. predictions[:, :max_k]). If there is at least one None,
        # sadly it means that we can't save any resource
        max_k = None
        all_ks = [metric.k for metric in metric_list]
        if None not in all_ks:
            max_k = max(all_ks)

        split_name = eval_dataset.split
        if split_name is None:
            print("WARNING: split name for the eval dataset passed is None. Fallback to 'eval'")
            split_name = "eval"
        else:
            split_name = str(split_name)

        preprocessed_eval = eval_dataset.map(
            self.rec_model.tokenize,
            remove_columns=eval_dataset.column_names,
            keep_in_memory=True,
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
            predictions, truths, loss = self.rec_model.generate_step(prepared_input)

            eval_loss += loss.item()

            total_preds.extend(predictions)
            total_truths.extend(truths)

            # we update the loss every 1% progress considering the total n° of batches
            # tqdm update integer percentage (1%, 2%) when float percentage is over .5 threshold (1.501 -> 2%)
            # so we print infos in the same way
            if round(100 * (i / total_n_batch)) > progress:
                result_so_far = self._compute_metrics(total_preds, total_truths, metric_list, max_k)

                pbar_desc = []

                if return_loss:
                    pbar_desc.append(f"{split_name} Loss -> {(eval_loss / i):.6f}")

                for metric_name, metric_val in result_so_far.items():
                    pbar_desc.append(f"{metric_name} -> {metric_val:.6f}")

                pbar_eval.set_description(", ".join(pbar_desc))

                progress += 1

        pbar_eval.close()

        eval_loss /= total_n_batch

        res_eval_dict = self._compute_metrics(total_preds, total_truths, metric_list, max_k)

        if return_loss is True:
            res_eval_dict[f"{split_name} loss"] = eval_loss

        return res_eval_dict

    @staticmethod
    def _compute_metrics(preds: List[np.ndarray], truths: List[np.ndarray], metric_list: List[Metric],
                         max_k: Optional[int]):

        # Pad array if necessary
        preds_so_far = PaddedArr(preds)
        truths_so_far = PaddedArr(truths)

        # Build rel binary matrix by cutting predictions to the max k desired
        # Save resources by not computing relevance for predictions outside the k range,
        # which are not used by any metric passed in input
        rel_binary_matrix = Metric.rel_binary_matrix(preds_so_far, truths_so_far, k=max_k)

        # when computing the specific metric result, we consider its k value which wil surely be <= max_k
        # (again, saving resources)
        result = {str(metric): metric(rel_binary_matrix[:, :metric.k])
                  for metric in metric_list}

        return result
