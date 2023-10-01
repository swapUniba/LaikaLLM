from math import ceil
from typing import List

import datasets
import numpy as np
from tqdm import tqdm

from src.evaluate.metrics import Metric
from src.model.t5 import T5FineTuned


class RecEvaluator:

    def __init__(self, rec_model: T5FineTuned, eval_batch_size: int):
        self.rec_model = rec_model
        self.eval_batch_size = eval_batch_size

    def evaluate(self, eval_dataset: datasets.Dataset, metric_list: List[Metric],
                 return_loss: bool = False):

        self.rec_model.eval()

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
            predictions, truths, loss = self.rec_model.valid_step(prepared_input)

            eval_loss += loss.item()

            total_preds.extend(predictions)
            total_truths.extend(truths)

            # we update the loss every 1% progress considering the total n° of batches
            # tqdm update integer percentage (1%, 2%) when float percentage is over .5 threshold (1.501 -> 2%)
            # so we print infos in the same way
            if round(100 * (i / total_n_batch)) > progress:
                preds_so_far = np.array(total_preds)
                truths_so_far = np.array(total_truths)

                result = {str(metric): metric(preds_so_far.squeeze(), truths_so_far)
                          for metric in metric_list}

                pbar_desc = []

                if return_loss:
                    pbar_desc.append(f"{split_name} Loss -> {(eval_loss / i):.6f}")

                for metric_name, metric_val in result.items():
                    pbar_desc.append(f"{metric_name} -> {metric_val:.6f}")

                pbar_eval.set_description(", ".join(pbar_desc))

                progress += 1

        pbar_eval.close()

        eval_loss /= total_n_batch

        total_preds = np.array(total_preds).squeeze()
        total_truths = np.array(total_truths)

        res_eval_dict = {str(metric): metric(total_preds, total_truths)
                         for metric in metric_list}

        if return_loss is True:
            res_eval_dict["loss"] = eval_loss

        return res_eval_dict
