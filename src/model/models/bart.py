from __future__ import annotations

import random
from typing import List, Literal

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast

from src.data.abstract_dataset import LaikaDataset
from src.data.abstract_templates import Task
from src.model.abstract_model import LaikaModel
from src.utils import dict_list2list_dict, list_dict2dict_list


class BartRecConfig(BartConfig):
    def __init__(self,
                 training_tasks_str: List[str] = None,
                 all_unique_labels: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.training_tasks_str = training_tasks_str
        self.all_unique_labels = all_unique_labels


class BartRec(LaikaModel, BartForConditionalGeneration):
    config_class = BartRecConfig

    def __init__(self,
                 config,
                 eval_task_str: str = None,
                 eval_template_id: int = None,
                 train_task_selection_strat: Literal['random', 'all'] = 'all'):

        BartForConditionalGeneration.__init__(self, config)

        LaikaModel.__init__(
            self,
            training_tasks_str=config.training_tasks_str,
            all_unique_labels=config.all_unique_labels,
            eval_task_str=eval_task_str,
            eval_template_id=eval_template_id,
            train_task_selection_strat=train_task_selection_strat
        )

        self.tokenizer = BartTokenizerFast.from_pretrained(config.name_or_path)

    @property
    def get_suggested_optimizer(self):
        return AdamW(
            list(self.parameters()),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

    def tokenize(self, batch):

        if "user_id" not in batch:
            raise AttributeError("This model expects 'user_id' column in the dataset to tokenize!")

        if not self.training and self.eval_task is None:
            raise ValueError("Model can't tokenize the eval task since no eval_task is set! "
                             "Pass it when initializing the model or with `set_eval_task()`")

        # from dict of lists to list of dicts
        batch = dict_list2list_dict(batch)

        encoded_sequence_list = []
        for sample in batch:

            if not self.training:
                tasks = [self.eval_task]
            elif self.train_task_selection_strat == "all":
                # Create a new shuffled list without modifying the original
                # we shuffle the train tasks to inject some randomness
                tasks = random.sample(self.training_tasks, len(self.training_tasks))
            else:
                tasks = [random.choice(self.training_tasks)]

            for task in tasks:
                # give all info that we have about the sample to the task randomly sampled to generate
                # input prompt and target text. Each task may have mandatory arguments, if they are missing
                # an assertion error will be raised
                templates_list = task(**sample)

                # each task gives as output a list: this list contains surely an inference prompt-target (i.e.,
                # a prompt target which could be used at inference time) and a variable number of support tasks
                # (i.e. tasks which do not have as target text the prediction of interest for the task)
                for (input_text, target_text, gt) in templates_list:

                    encoded_sequence = self.tokenizer(text=input_text, text_target=target_text, truncation=True)

                    # get word ids from t5 tokenizer fast
                    whole_word_ids = np.array(encoded_sequence.encodings[0].word_ids)
                    special_token_mask = np.array(encoded_sequence.encodings[0].special_tokens_mask).astype(bool)

                    # we set -1 to all special tokens (to substitute None, which is the value set by default)
                    whole_word_ids[~special_token_mask] += 1
                    whole_word_ids[special_token_mask] = self.tokenizer.pad_token_id
                    encoded_sequence["whole_word_ids"] = whole_word_ids.tolist()

                    if not self.training:

                        if gt is None:
                            raise ValueError("In the __call__ method of the template, the `gt` attribute should be "
                                             "set for templates used in the evaluation phase!")

                        # it may be the item id or the item rating for example, depending on the task chosen
                        encoded_sequence["gt"] = gt

                    encoded_sequence_list.append(encoded_sequence)

        # from list of dicts to dict of lists
        return list_dict2dict_list(encoded_sequence_list)

    def prepare_input(self, batch):
        input_dict = {}

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"],
                                      batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
        whole_word_ids = pad_sequence(batch["whole_word_ids"],
                                      batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)

        input_dict["input_ids"] = input_ids.to(self.device)
        input_dict["attention_mask"] = attention_mask.to(self.device)
        input_dict["whole_word_ids"] = whole_word_ids.to(self.device)

        if "labels" in batch:
            lm_labels = pad_sequence(batch["labels"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

            input_dict["labels"] = lm_labels.to(self.device)

        if "gt" in batch:
            input_dict["gt"] = batch["gt"]

        return input_dict

    def train_step(self, batch):

        output = self(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"])

        return output.loss

    @torch.no_grad()
    def generate_step(self, batch):

        if self.eval_task is None:
            raise ValueError("Model can't perform generate_step since no eval_task is set! "
                             "Pass it when initializing the model or with `set_eval_task()`")

        # if it's not a ranking task (e.g., it is a rating prediction task),
        # we should return one prediction for ground truth element.
        # In theory there could be a complex way of mixing multiple text generated into a single prediction
        # (e.g., avg of 10 rating predictions), but here we simply reduce the num return sequences
        num_return_sequences = 10 if self.eval_task.is_ranking_task else 1
        max_new_tokens = 50
        num_beams = 30
        no_repeat_ngram_size = 0
        early_stopping = True

        gt = np.array(batch.pop("gt"))

        output = self(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"])

        beam_outputs = self.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping
        )

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        mapped_predictions = np.array(generated_sents).reshape((len(gt), num_return_sequences))

        loss = output.loss

        return mapped_predictions, gt, loss

    def save(self, output_dir: str):

        # save hf model and parameters that we added to the config
        self.save_pretrained(save_directory=output_dir)

        # also tokenizer is saved
        self.tokenizer.save_pretrained(save_directory=output_dir)

    @classmethod
    def load(cls, dir_path: str, **kwargs):
        return cls.from_pretrained(dir_path, **kwargs)

    def train(self, mode: bool = True):

        if mode is True:
            Task.train()
        else:
            Task.eval()

        return BartForConditionalGeneration.train(self, mode)

    def to(self, device: str):
        return BartForConditionalGeneration.to(self, device)

    @classmethod
    def from_cls(cls, model_cls: type[BartRec], dataset_obj: LaikaDataset, **kwargs):

        kwargs["all_unique_labels"] = dataset_obj.all_items.tolist()

        return model_cls.from_pretrained(**kwargs)
