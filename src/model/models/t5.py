from __future__ import annotations

import random
import re
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, Adafactor, T5TokenizerFast, T5Config

from src.data.abstract_dataset import LaikaDataset
from src.data.templates.templates import Task
from src.model.abstract_model import LaikaModel
from src.utils import dict_list2list_dict, list_dict2dict_list


class UserEmbeds(nn.Module):

    def __init__(self, n_users, dim_model):
        super().__init__()

        self.emb_layer = nn.Embedding(n_users, dim_model)
        torch.nn.init.xavier_uniform_(self.emb_layer.weight)

    def __call__(self, user_idx):
        x = self.emb_layer(user_idx)

        # we dropout an entire column (neuron)
        x = x.permute(1, 0)
        x = nn.functional.dropout1d(x, p=0.6, training=self.training)
        x = x.permute(1, 0)

        x = nn.functional.leaky_relu(x)

        return x


class T5RecConfig(T5Config):
    def __init__(self,
                 training_tasks_str: List[str] = None,
                 n_users: int = None,
                 all_unique_labels: List[str] = None,
                 inject_personalization: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.training_tasks_str = training_tasks_str
        self.n_users = n_users
        self.all_unique_labels = all_unique_labels
        self.inject_personalization = inject_personalization


class T5Rec(LaikaModel, T5ForConditionalGeneration):
    config_class = T5RecConfig

    def __init__(self,
                 config,
                 eval_task_str: str = None,
                 eval_template_id: int = None):

        T5ForConditionalGeneration.__init__(self, config)

        LaikaModel.__init__(
            self,
            training_tasks_str=config.training_tasks_str,
            all_unique_labels=config.all_unique_labels,
            eval_task_str=eval_task_str,
            eval_template_id=eval_template_id
        )

        self.tokenizer = T5TokenizerFast.from_pretrained(config.name_or_path)

        if config.inject_personalization is True:

            if config.n_users is None:
                raise AttributeError("n_users parameter can't be None when "
                                     "inject_personalization is True!")

            self.user_embeddings = UserEmbeds(config.n_users, self.config.d_model)

    @property
    def get_suggested_optimizer(self):
        return Adafactor(
            list(self.parameters()),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.01,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

    def tokenize(self, batch):

        if "user_id" not in batch or "gt_item" not in batch:
            raise AttributeError("This model expects 'user_id' and 'gt_item' columns in the dataset to tokenize!")

        # from dict of lists to list of dicts
        batch = dict_list2list_dict(batch)

        encoded_sequence_list = []
        for sample in batch:

            task = random.choice(self.training_tasks) if self.training else self.eval_task

            # give all info that we have about the sample to the task randomly sampled to generate
            # input prompt and target text. Each task may have mandatory arguments, if they are missing
            # an assertion error will be raised
            templates_list = task(**sample)

            # TO DO: make example that works for split different from leave one out
            for input_text, target_text in templates_list:
                encoded_sequence = self.tokenizer(text=input_text, text_target=target_text, truncation=True)

                # get word ids from t5 tokenizer fast
                whole_word_ids = np.array(encoded_sequence.encodings[0].word_ids)
                special_token_mask = np.array(encoded_sequence.encodings[0].special_tokens_mask).astype(bool)

                # we set -1 to all special tokens (to substitute None, which is the value set by default)
                whole_word_ids[~special_token_mask] += 1
                whole_word_ids[special_token_mask] = self.tokenizer.pad_token_id

                # even if surely there is only one user, we must wrap it into a list for the batched map fn to work
                encoded_sequence["user_idx"] = [int(re.search(r"\d+", sample["user_id"]).group())]
                encoded_sequence["whole_word_ids"] = whole_word_ids.tolist()
                encoded_sequence["gt_item"] = sample["gt_item"]

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

        input_dict["user_idx"] = batch["user_idx"].to(self.device)
        input_dict["input_ids"] = input_ids.to(self.device)
        input_dict["attention_mask"] = attention_mask.to(self.device)
        input_dict["whole_word_ids"] = whole_word_ids.to(self.device)

        if "labels" in batch:
            lm_labels = pad_sequence(batch["labels"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

            input_dict["labels"] = lm_labels.to(self.device)

        if not self.training:
            input_dict["gt_item"] = batch["gt_item"]

        return input_dict

    def _inject_personalization(self, token_inputs_embeds: Tensor, user_idxs: Tensor):

        # whole_word_embeds = self.whole_word_embeddings(whole_word_ids)
        # # whole_word_embeds = self.relu(whole_word_embeds)
        # assert whole_word_embeds.shape[-1] == token_inputs_embeds.shape[-1]
        # inputs_embeds = token_inputs_embeds + whole_word_embeds

        # user idxs start from 1, TO IMPROVE!
        user_embeds = self.user_embeddings(user_idxs - 1).unsqueeze(dim=1)
        # whole_word_embeds = self.relu(whole_word_embeds)
        inputs_embeds = token_inputs_embeds + user_embeds

        return inputs_embeds

    def train_step(self, batch):

        inputs_embeds = self.shared(batch["input_ids"])

        if self.config.inject_personalization is True:
            inputs_embeds = self._inject_personalization(inputs_embeds, batch["user_idx"].squeeze())

        output = self(inputs_embeds=inputs_embeds,
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"])

        return output.loss

    @torch.no_grad()
    def generate_step(self, batch):

        if self.eval_task is None:
            raise ValueError("Model can't perform valid_step since no eval_task is set! "
                             "Pass it when initializing the model or with `set_eval_task()`")

        num_return_sequences = 10
        max_new_tokens = 50
        num_beams = 30
        no_repeat_ngram_size = 0
        early_stopping = True

        gt_items = batch.pop("gt_item")

        inputs_embeds = self.shared(batch["input_ids"])
        if self.config.inject_personalization is True:
            inputs_embeds = self._inject_personalization(inputs_embeds, batch["user_idx"].squeeze())

        output = self(inputs_embeds=inputs_embeds,
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"])

        beam_outputs = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping
        )

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

        mapped_predictions = np.array(generated_sents).reshape((len(gt_items), num_return_sequences))
        val_loss = output.loss

        return mapped_predictions, gt_items, val_loss

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

        return T5ForConditionalGeneration.train(self, mode)

    def to(self, device: str):
        return T5ForConditionalGeneration.to(self, device)

    @classmethod
    def from_automatic_usage(cls, dataset_obj: LaikaDataset, **kwargs):

        kwargs["all_unique_labels"] = dataset_obj.all_items.tolist()
        kwargs["n_users"] = len(dataset_obj.all_users)

        return cls.from_pretrained(**kwargs)
