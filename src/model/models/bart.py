from __future__ import annotations

import random
from copy import deepcopy
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast, GPT2Config, GPT2LMHeadModel, \
    GPT2TokenizerFast, OPTForCausalLM, AutoTokenizer, OPTConfig

from src.data.abstract_dataset import LaikaDataset
from src.data.abstract_templates import Task
from src.model.abstract_model import LaikaModel
from src.utils import dict_list2list_dict, list_dict2dict_list


class GPT2RecConfig(GPT2Config):
    def __init__(self,
                 training_tasks_str: List[str] = None,
                 all_unique_labels: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.training_tasks_str = training_tasks_str
        self.all_unique_labels = all_unique_labels


class GPT2Rec(LaikaModel, GPT2LMHeadModel):
    config_class = GPT2RecConfig

    def __init__(self,
                 config,
                 eval_task_str: str = None,
                 eval_template_id: int = None,
                 train_task_selection_strat: Literal['random', 'all'] = 'all'):

        GPT2LMHeadModel.__init__(self, config)

        LaikaModel.__init__(
            self,
            training_tasks_str=config.training_tasks_str,
            all_unique_labels=config.all_unique_labels,
            eval_task_str=eval_task_str,
            eval_template_id=eval_template_id,
            train_task_selection_strat=train_task_selection_strat
        )

        self.tokenizer = GPT2TokenizerFast.from_pretrained(config.name_or_path)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    @property
    def get_suggested_optimizer(self):
        return AdamW(
            list(self.parameters()),
            lr=5e-5,
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
                templates_list = task(catalog_items=self.all_unique_labels, **sample)

                # each task gives as output a list: this list contains surely an inference prompt-target (i.e.,
                # a prompt target which could be used at inference time) and a variable number of support tasks
                # (i.e. tasks which do not have as target text the prediction of interest for the task)
                for (input_text, target_text, gt) in templates_list:

                    target_ids = self.tokenizer(target_text, truncation=True).input_ids

                    input_text_ids = self.tokenizer(f"Input: {input_text} \nTarget: ",
                                                    truncation=True,
                                                    max_length=self.tokenizer.model_max_length - len(target_ids) - 1).input_ids

                    encoded_sequence: dict = {
                        "input_prompt_ids": input_text_ids,
                        "target_prompt_ids": target_ids,
                        "attention_mask_prompt": [1] * len(input_text_ids),
                        "input_ids": input_text_ids + target_ids,
                        "attention_mask": [1] * (len(input_text_ids) + len(target_ids))
                    }

                    encoded_sequence["labels"] = deepcopy(encoded_sequence["input_ids"])

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

        # decoder only model should be padded to the left when performing batch inference,
        # otherwise you are continuing generating over a pad token which was not observed during training.
        # Check https://github.com/huggingface/transformers/issues/3021#issuecomment-1454267951
        # and https://github.com/kzl/decision-transformer/issues/36
        input_ids = pad_sequence([a.flip(dims=[0]) for a in batch["input_ids"]],
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
        attention_mask = pad_sequence([a.flip(dims=[0]) for a in batch["attention_mask"]],
                                      batch_first=True,
                                      padding_value=0).flip(dims=[1])

        input_prompt_ids = pad_sequence([a.flip(dims=[0]) for a in batch["input_prompt_ids"]],
                                        batch_first=True,
                                        padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
        attention_mask_prompt = pad_sequence([a.flip(dims=[0]) for a in batch["attention_mask_prompt"]],
                                             batch_first=True,
                                             padding_value=0).flip(dims=[1])

        input_ids = torch.hstack((input_ids, torch.full((input_ids.shape[0], 1), fill_value=self.tokenizer.eos_token_id)))
        attention_mask = torch.hstack(
            (attention_mask, torch.full((input_ids.shape[0], 1), fill_value=1)))

        input_dict["input_ids"] = input_ids.to(self.device)
        input_dict["attention_mask"] = attention_mask.to(self.device)

        input_dict["input_ids_prompt"] = input_prompt_ids.to(self.device)
        input_dict["attention_mask_prompt"] = attention_mask_prompt.to(self.device)

        if "labels" in batch:
            lm_labels = pad_sequence([a.flip(dims=[0]) for a in batch["labels"]],
                                     batch_first=True,
                                     padding_value=-100).flip(dims=[1])

            lm_labels = torch.hstack(
                (lm_labels, torch.full((input_ids.shape[0], 1), fill_value=self.tokenizer.eos_token_id)))

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
        #
        # output = self(input_ids=batch["input_ids"],
        #               attention_mask=batch["attention_mask"],
        #               labels=batch["labels"])

        beam_outputs = self.generate(
            input_ids=batch["input_ids_prompt"],
            attention_mask=batch["attention_mask_prompt"],
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping
        )

        # this works for all because when generating, also pad tokens are generated!
        beam_outputs_targets = beam_outputs[:, batch["input_ids_prompt"].shape[1]:]

        generated_sents = self.tokenizer.batch_decode(beam_outputs_targets, skip_special_tokens=True)
        mapped_predictions = np.array(generated_sents).reshape((len(gt), num_return_sequences))

        return mapped_predictions, gt, torch.tensor(0)

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

        return GPT2LMHeadModel.train(self, mode)

    def to(self, device: str):
        return GPT2LMHeadModel.to(self, device)

    @classmethod
    def from_cls(cls, model_cls: type[GPT2Rec], dataset_obj: LaikaDataset, **kwargs):

        kwargs["all_unique_labels"] = dataset_obj.all_items.tolist()

        return model_cls.from_pretrained(**kwargs)
