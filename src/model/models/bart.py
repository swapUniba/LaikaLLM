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

        total_input_ids = pad_sequence(
            batch["total_input_ids"],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        total_attention_mask = pad_sequence(
            batch["total_attention_mask"],
            batch_first=True,
            padding_value=0
        )

        input_prompt_ids = pad_sequence(
            batch["input_prompt_ids"],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        input_prompt_attention_mask = pad_sequence(
            batch["input_prompt_attention_mask"],
            batch_first=True,
            padding_value=0
        )

        # pad value to -100 so that it's ignored when computing cross entropy
        lm_labels = pad_sequence(
            batch["total_labels"],
            batch_first=True,
            padding_value=-100
        )

        input_dict["total_input_ids"] = total_input_ids.to(self.device)
        input_dict["total_attention_mask"] = total_attention_mask.to(self.device)

        input_dict["input_prompt_ids"] = input_prompt_ids.to(self.device)
        input_dict["input_prompt_attention_mask"] = input_prompt_attention_mask.to(self.device)
        input_dict["total_labels"] = lm_labels.to(self.device)

        if "gt" in batch:
            input_dict["gt"] = batch["gt"]

        return input_dict

    def train_step(self, batch):

        output = self(input_ids=batch["total_input_ids"],
                      attention_mask=batch["total_attention_mask"],
                      labels=batch["total_labels"])

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
        num_beams = 30
        no_repeat_ngram_size = 0
        early_stopping = True

        gt = np.array(batch.pop("gt"))
        #
        # output = self(input_ids=batch["input_ids"],
        #               attention_mask=batch["attention_mask"],
        #               labels=batch["labels"])

        # for decoder only model, input should be padded to the left when performing batch inference with generate,
        # otherwise you are continuing generating over a pad token which was not observed during training!
        # Check https://github.com/huggingface/transformers/issues/3021#issuecomment-1454267951
        left_padded_input_ids, left_padded_attn_mask = self._left_pad(batch["input_prompt_ids"],
                                                                      batch["input_prompt_attention_mask"])

        beam_outputs = self.generate(
            input_ids=left_padded_input_ids,
            attention_mask=left_padded_attn_mask,
            num_return_sequences=num_return_sequences,
            max_length=self.tokenizer.model_max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping
        )

        # this works for all rows of tensor because when generating also pad tokens are generated,
        # so length to "cut" is in common to all rows!
        beam_outputs_targets = beam_outputs[:, batch["input_prompt_ids"].shape[1]:]

        generated_sents = self.tokenizer.batch_decode(beam_outputs_targets, skip_special_tokens=True)
        mapped_predictions = np.array(generated_sents).reshape((len(gt), num_return_sequences))

        return mapped_predictions, gt, torch.tensor(0)

    def _left_pad(self, input_prompt_ids: torch.LongTensor, input_prompt_attention_mask: torch.LongTensor):

        # calculate the number of padding tokens in each row, which is where
        # the valid ids start for each row
        num_padding_tokens = (input_prompt_ids == self.tokenizer.pad_token_id).sum(dim=1)

        # create a new tensor filled with padding tokens (zero tokens for attention mask)
        left_padded_input_ids = torch.full_like(input_prompt_ids, fill_value=self.tokenizer.pad_token_id)
        left_padded_attn_mask = torch.zeros_like(input_prompt_attention_mask)

        # For each row, move the actual content to the specified positions
        for i in range(input_prompt_ids.size(0)):

            end_index_valid_ids = input_prompt_ids.size(1) - num_padding_tokens[i]

            left_padded_input_ids[i, num_padding_tokens[i]:] = input_prompt_ids[i, :end_index_valid_ids]
            left_padded_attn_mask[i, num_padding_tokens[i]:] = 1

        return left_padded_input_ids, left_padded_attn_mask

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
