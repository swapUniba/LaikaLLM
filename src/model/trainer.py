import os
import time
from math import ceil
from typing import Optional, Literal, Callable, Dict

import datasets
import numpy as np
from tqdm import tqdm

from src import MODELS_DIR, ExperimentConfig
from src.utils import log_wandb
from src.data.amazon_dataset import AmazonDataset
from src.data.templates import SequentialTask
from src.evaluate.metrics import Metric, Accuracy, Hit
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
                "train/loss": train_loss
            })

            pbar.close()

            if validation_dataset is not None:
                val_result = self.validation(validation_dataset=validation_dataset)

                if self.monitor_strategy == "loss":
                    monitor_str = "Val loss"
                    monitor_val = val_result["loss"]
                    should_save = monitor_val < best_val_monitor_result  # we want loss to decrease
                else:
                    metric_obj, monitor_val = val_result["metric"]
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

    def validation(self, validation_dataset: datasets.Dataset):

        print("VALIDATION")
        self.rec_model.eval()

        preprocessed_val = validation_dataset.map(
            self.rec_model.tokenize,
            remove_columns=validation_dataset.column_names,
            keep_in_memory=True,
            desc="Tokenizing val set"
        )
        preprocessed_val.set_format("torch")

        # ceil because we don't drop the last batch
        total_n_batch = ceil(preprocessed_val.num_rows / self.eval_batch_size)

        pbar_val = tqdm(preprocessed_val.iter(batch_size=self.eval_batch_size),
                        total=total_n_batch)

        metric: Metric = Accuracy()
        val_loss = 0
        total_preds = []
        total_truths = []

        # progress will go from 0 to 100. Init to -1 so at 0 we perform the first print
        progress = -1
        for i, batch in enumerate(pbar_val, start=1):

            prepared_input = self.rec_model.prepare_input(batch)
            predictions, truths, loss = self.rec_model.valid_step(prepared_input)

            val_loss += loss.item()

            total_preds.extend(predictions)
            total_truths.extend(truths)

            # we update the loss every 1% progress considering the total n° of batches
            # tqdm update integer percentage (1%, 2%) when float percentage is over .5 threshold (1.501 -> 2%)
            # so we print infos in the same way
            if round(100 * (i / total_n_batch)) > progress:
                preds_so_far = np.array(total_preds)
                truths_so_far = np.array(total_truths)

                if len(preds_so_far.squeeze().shape) > 1:
                    metric = Hit()

                result = metric(preds_so_far.squeeze(), truths_so_far)
                pbar_val.set_description(f"Val Loss -> {(val_loss / i):.6f}, "
                                         f"{metric} -> {result:.3f}")

                progress += 1

        pbar_val.close()

        val_loss /= total_n_batch
        val_metric = metric(np.array(total_preds).squeeze(), np.array(total_truths))

        # val_loss is computed for the entire batch, not for each sample, that's why is safe
        # to use pbar_val
        return {"loss": val_loss, "metric": (metric, val_metric)}


def trainer_main():

    n_epochs = ExperimentConfig.n_epochs
    batch_size = ExperimentConfig.train_batch_size
    eval_batch_size = ExperimentConfig.eval_batch_size
    device = ExperimentConfig.device
    checkpoint = ExperimentConfig.checkpoint
    random_seed = ExperimentConfig.random_seed

    ds = AmazonDataset(dataset_name="toys")

    ds_dict = ds.get_hf_datasets()
    all_unique_labels = ds.all_items

    train = ds_dict["train"]
    val = ds_dict["validation"]

    # REDUCE FOR TESTING
    # train = Dataset.from_dict(train[:5000])
    # val = Dataset.from_dict(val[:5000])

    sampling_fn = ds.sample_train_sequence

    train_task_list = [SequentialTask()]

    rec_model = T5FineTuned.from_pretrained(
        checkpoint,
        training_tasks=train_task_list,
        all_unique_labels=all_unique_labels,
        device=device
    )

    # new_words = ['<']
    #
    # model_ntp.tokenizer.add_tokens(new_words)
    # model_ntp.model.resize_token_embeddings(len(model_ntp.tokenizer))

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

    # validation only at last epoch
    trainer.train(train)

    for template_id in SequentialTask.templates_dict.keys():

        print(f"Validating on {template_id}")

        rec_model.set_eval_task(SequentialTask(force_template_id=template_id))
        res = trainer.validation(val)

        log_wandb({
            "val/template": template_id,
            "val/hit@10": res["metric"],
            "val/loss": res["loss"],
        })


if __name__ == "__main__":
    trainer_main()
