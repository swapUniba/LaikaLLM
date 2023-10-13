import os
import time
from collections import defaultdict
from math import ceil
from typing import Optional, Literal, Callable, Dict

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
from src.evaluate.metrics import Hit
from src.model.t5 import T5FineTuned


class RecTrainer:

    def __init__(self,
                 n_epochs: int,
                 batch_size: int,
                 rec_model,
                 all_labels: np.ndarray,
                 train_sampling_fn: Callable[[Dict], Dict],
                 device: str = 'cuda:0',
                 monitor_strategy: Literal['loss', 'metric'] = 'metric',
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
        self.monitor_strategy = monitor_strategy

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

        # depending on the monitor strategy, we want either this to decrease or to increase,
        # so we have a different initialization
        best_val_monitor_result = np.inf if self.monitor_strategy == "loss" else 0
        best_epoch = -1

        optimizer = self.rec_model.get_suggested_optimizer()

        start = time.time()
        for epoch in range(0, self.n_epochs):

            # at the start of each iteration, we randomly sample the train sequence and tokenize it
            shuffled_train = train_dataset.shuffle(seed=self.random_seed)
            sampled_train = shuffled_train.map(self.train_sampling_fn,
                                               remove_columns=train_dataset.column_names,
                                               keep_in_memory=True,
                                               desc="Sampling train set")

            preprocessed_train = sampled_train.map(self.rec_model.tokenize,
                                                   remove_columns=sampled_train.column_names,
                                                   load_from_cache_file=False,
                                                   keep_in_memory=True,
                                                   desc="Tokenizing train set")
            preprocessed_train.set_format("torch")

            # ceil because we don't drop the last batch. It's here since if we are in
            # augment strategy, row number increases after preprocessing
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

            log_wandb({
                "train/loss": train_loss,
                "train/epoch": epoch + 1
            })

            pbar.close()

            if validation_dataset is not None:

                validation_metric = Hit(k=10)

                val_result = self.rec_evaluator.evaluate(validation_dataset,
                                                         metric_list=[validation_metric],
                                                         return_loss=True)

                if self.monitor_strategy == "loss":
                    monitor_str = "Val loss"
                    monitor_val = val_result["loss"]
                    should_save = monitor_val < best_val_monitor_result  # we want loss to decrease
                else:
                    metric_obj, monitor_val = val_result[str(validation_metric)]
                    monitor_str = str(metric_obj)
                    should_save = monitor_val > best_val_monitor_result  # we want metric (acc/hit) to increase

                # we save the best model based on the reference metric result
                if should_save:
                    best_epoch = epoch + 1  # we start from 0
                    best_val_monitor_result = monitor_val
                    self.rec_model.save(self.output_path)

                    print(f"{monitor_str} improved, model saved into {self.output_path}!")
            else:
                self.rec_model.save_pretrained(self.output_path)

        elapsed_time = (time.time() - start) / 60
        print(" Train completed! Check models saved into 'models' dir ".center(100, "*"))
        print(f"Time -> {elapsed_time}")
        print(f"Best epoch -> {best_epoch}")

        log_wandb({
            "train/elapsed_time": elapsed_time,
            "train/best_epoch": best_epoch
        })


def trainer_main():

    n_epochs = ExperimentConfig.n_epochs
    batch_size = ExperimentConfig.train_batch_size
    eval_batch_size = ExperimentConfig.eval_batch_size
    device = ExperimentConfig.device
    checkpoint = ExperimentConfig.checkpoint
    random_seed = ExperimentConfig.random_seed
    train_tasks = ExperimentConfig.train_tasks

    ds = AmazonDataset.load()

    ds_dict = ds.get_hf_datasets()
    all_unique_labels = ds.all_items
    all_unique_users = ds.all_users

    train = ds_dict["train"]
    val = ds_dict["validation"]
    test = ds_dict["test"]

    # REDUCE FOR TESTING
    # train = Dataset.from_dict(train[:5000])
    # val = Dataset.from_dict(val[:5000])
    # test = Dataset.from_dict(test[:100])

    sampling_fn = ds.sample_train_sequence

    # from strings to objects initialized
    train_task_list = Task.from_string(*train_tasks)

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
        device=device
    )

    trainer = RecTrainer(
        rec_model=rec_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        random_seed=random_seed,
        train_sampling_fn=sampling_fn,
        monitor_strategy="loss",
        output_name=ExperimentConfig.exp_name
    )

    # no validation at the moment
    trainer.train(train)

    # eval
    evaluator = RecEvaluator(rec_model, eval_batch_size)
    metric_list = [Hit(k=10), Hit(k=5)]
    cumulative_results = defaultdict(list)
    for task in train_task_list:
        for template_id in task.templates_dict.keys():

            print(f"Evaluating on {task}/{template_id}")
            task.force_template(template_id)
            rec_model.set_eval_task(task)

            res = evaluator.evaluate(test, metric_list=metric_list)

            dict_to_log = {f"test/{task}/template_id": template_id}
            for metric_name, metric_val in res.items():
                dict_to_log[f"test/{task}/{metric_name}"] = metric_val
                cumulative_results[str(metric_name)].append(metric_val)

            log_wandb(dict_to_log)

        average_results = {metric: np.mean(cumulative_metric_result).item()
                           for metric, cumulative_metric_result in cumulative_results.items()}

        log_wandb({f"test/{task}/avg_results/{metric}": avg_metric_result
                   for metric, avg_metric_result in average_results.items()})

        print(f"Average result for task {task}:")
        print(average_results)


if __name__ == "__main__":
    trainer_main()
