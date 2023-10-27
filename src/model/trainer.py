import os
import time
from math import ceil
from typing import Optional, Callable, Dict

import datasets
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from src import MODELS_DIR, ExperimentConfig
from src.evaluate.evaluator import RecEvaluator
from src.utils import log_wandb
from src.data.amazon_dataset import AmazonDataset
from src.data.templates import Task
from src.evaluate.metrics import Metric
from src.model.t5 import T5Rec


class RecTrainer:

    def __init__(self,
                 n_epochs: int,
                 batch_size: int,
                 rec_model,
                 all_labels: np.ndarray,
                 train_sampling_fn: Callable[[Dict], Dict],
                 device: str = 'cuda:0',
                 monitor_metric: str = 'loss',
                 eval_batch_size: Optional[int] = None,
                 output_name: Optional[str] = None,
                 random_seed: Optional[int] = None):

        self.rec_model = rec_model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.all_labels = all_labels
        self.train_sampling_fn = train_sampling_fn
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.device = device
        self.random_seed = random_seed

        if monitor_metric != "loss":
            monitor_metric = RankingMetric.from_string(monitor_metric)

        self.monitor_metric = monitor_metric

        # output name
        if output_name is None:
            # replace '/' with '_' to avoid creation of subdir (google/flan-t5-small -> google_flan-t5-small)
            output_name = f"{rec_model.config.name_or_path.replace('/', '_')}_{n_epochs}"

        self.output_name = output_name
        self.output_path = os.path.join(MODELS_DIR, output_name)

        # evaluator for validating with validation set during training
        self.rec_evaluator = RecEvaluator(self.rec_model, self.eval_batch_size)

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        self.rec_model.train()

        # depending on the monitor metric, we want either this to decrease or to increase,
        # so we have a different initialization
        best_val_monitor_result = np.inf if self.monitor_metric == "loss" else 0
        best_epoch = -1

        optimizer = self.rec_model.get_suggested_optimizer()

        start = time.time()
        for epoch in range(0, self.n_epochs):

            self.rec_model.train()

            # at the start of each iteration, we randomly sample the train sequence and tokenize it
            # batched set to True because data can be augmented, either when sampling or when
            # tokenizing (e.g. a task as multiple support templates)

            sampled_train = train_dataset.map(self.train_sampling_fn,
                                              remove_columns=train_dataset.column_names,
                                              keep_in_memory=True,
                                              batched=True,
                                              desc="Sampling train set")

            preprocessed_train = sampled_train.map(self.rec_model.tokenize,
                                                   remove_columns=sampled_train.column_names,
                                                   keep_in_memory=True,
                                                   batched=True,
                                                   desc="Tokenizing train set")

            # shuffle here so that if we augment data (2 or more row for a single user) it is shuffled
            preprocessed_train = preprocessed_train.shuffle()
            preprocessed_train.set_format("torch")

            # ceil because we don't drop the last batch
            total_n_batch = ceil(preprocessed_train.num_rows / self.batch_size)

            pbar = tqdm(preprocessed_train.iter(batch_size=self.batch_size),
                        total=total_n_batch)

            train_loss = 0

            # progress will go from 0 to 100. Init to -1 so at 0 we perform the first print
            progress = -1
            for i, batch in enumerate(pbar, start=1):

                optimizer.zero_grad()

                prepared_input = self.rec_model.prepare_input(batch)
                loss = self.rec_model.train_step(prepared_input)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # we update the loss every 1% progress considering the total n° of batches
                # tqdm update integer percentage (1%, 2%) when float percentage is over .5 threshold (1.501 -> 2%)
                # so we print infos in the same way
                if round(100 * (i / total_n_batch)) > progress:
                    pbar.set_description(f"Epoch {epoch + 1}, Loss -> {(train_loss / i):.6f}")
                    progress += 1
                    log_wandb({
                        "train/loss": train_loss / i
                    })

            train_loss /= total_n_batch

            pbar.close()

            dict_to_log = {
                "train/loss": train_loss,
                "train/epoch": epoch + 1
            }

            if validation_dataset is not None:

                self.rec_model.eval()

                # monitor_metric is either the string "loss" or a RankingMetric obj
                validation_metric = self.monitor_metric

                val_result = self.rec_evaluator.evaluate(validation_dataset,
                                                         metric_list=[validation_metric],
                                                         return_loss=True)

                if validation_metric == "loss":
                    monitor_str = "Val loss"
                    monitor_val = val_result["loss"]

                    # we want loss to decrease
                    should_save = monitor_val < best_val_monitor_result
                else:
                    monitor_str = str(validation_metric)
                    monitor_val = val_result[monitor_str]

                    # The validation metric chosen should decide how to compare best result
                    # (mse should decrease, hit should increase, etc.)
                    should_save = validation_metric.operator_comparison(monitor_val, best_val_monitor_result)

                # we save the best model based on the metric/loss result
                if should_save:
                    best_epoch = epoch + 1  # we start from 0
                    best_val_monitor_result = monitor_val
                    self.rec_model.save_pretrained(self.output_path)

                    print(f"{monitor_str} improved, model saved into {self.output_path}!")

                # prefix "val" for val result dict
                val_to_log = {f"val/{metric_name}": metric_val for metric_name, metric_val in val_result.items()}
                val_to_log["val/epoch"] = epoch + 1

                dict_to_log.update(val_to_log)

            else:
                self.rec_model.save_pretrained(self.output_path)

            # log to wandb at each epoch
            log_wandb(dict_to_log)

        elapsed_time = (time.time() - start) / 60
        print(" Train completed! Check models saved into 'models' dir ".center(100, "*"))
        print(f"Time -> {elapsed_time}")
        print(f"Best epoch -> {best_epoch}")

        log_wandb({
            "train/elapsed_time": elapsed_time,
            "train/best_epoch": best_epoch
        })

        # return best model path if validation was set, otherwise this return the path of the model
        # saved at the last epoch
        return self.output_path


def trainer_main():
    exp_name = ExperimentConfig.exp_name
    n_epochs = ExperimentConfig.n_epochs
    batch_size = ExperimentConfig.train_batch_size
    eval_batch_size = ExperimentConfig.eval_batch_size
    device = ExperimentConfig.device
    checkpoint = ExperimentConfig.checkpoint
    random_seed = ExperimentConfig.random_seed
    train_tasks = ExperimentConfig.train_tasks
    monitor_metric = ExperimentConfig.monitor_strategy

    ds = AmazonDataset.load()

    ds_dict = ds.get_hf_datasets()
    all_unique_labels = ds.all_items
    all_unique_users = ds.all_users

    train = ds_dict["train"]
    val = ds_dict["validation"] if monitor_metric != "no" else None
    test = ds_dict["test"]

    # REDUCE FOR TESTING
    # train = Dataset.from_dict(train[:5000])
    # val = Dataset.from_dict(val[:5000])
    # test = Dataset.from_dict(test[:100])

    sampling_fn = ds.sample_train_sequence

    # from strings to objects initialized
    train_task_list = Task.from_string(*train_tasks, all_unique_items=all_unique_labels)

    # Log all templates used
    dataframe_dict = {"task_type": [], "template_id": [], "input_prompt": [], "target_text": []}
    for task in train_task_list:
        for template_id in task.templates_dict:
            input_prompt, target_text = task.templates_dict[template_id]

            dataframe_dict["task_type"].append(str(task))
            dataframe_dict["template_id"].append(template_id)
            dataframe_dict["input_prompt"].append(input_prompt)
            dataframe_dict["target_text"].append(target_text)

    dataframe = pd.DataFrame(dataframe_dict)

    log_wandb({"task_templates": wandb.Table(dataframe=dataframe)})

    rec_model = T5FineTuned.from_pretrained(
        checkpoint,
        n_users=len(all_unique_users),
        training_tasks=train_task_list,
        all_unique_labels=all_unique_labels,
        device=device,
        eval_task=deepcopy(train_task_list[0]).force_template(0)  # validation task
    )

    trainer = RecTrainer(
        rec_model=rec_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        random_seed=random_seed,
        train_sampling_fn=sampling_fn,
        monitor_metric=monitor_metric,
        output_name=exp_name
    )

    trainer.train(train, validation_dataset=val)


if __name__ == "__main__":
    ExperimentConfig.content_indexing = True
    ExperimentConfig.inject_personalization = ("train", "eval")
    ExperimentConfig.train_tasks = ("SequentialSideInfoTask",)

    trainer_main()
