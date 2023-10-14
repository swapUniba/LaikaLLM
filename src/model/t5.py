from __future__ import annotations
import random
import re
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, Adafactor, T5TokenizerFast
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from src import ExperimentConfig
from src.data.templates import Task

# sim_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda:0")


class T5FineTuned(T5ForConditionalGeneration):

    def __init__(self,
                 config,
                 n_users: int,
                 training_tasks: List[Task],
                 all_unique_labels: np.ndarray[str],
                 eval_task: Task = None,
                 device: str = "cpu"):

        super().__init__(config)

        self.tokenizer = T5TokenizerFast.from_pretrained(config.name_or_path)

        self.training_tasks = training_tasks
        self.eval_task = eval_task

        self.n_users = n_users
        self.all_unique_labels = all_unique_labels
        # self.encoded_all_labels = sim_model.encode(list(self.all_unique_labels),
        #                                            convert_to_tensor=True,
        #                                            show_progress_bar=True)

        # Set maximum 512 whole words in a source text
        self.user_embeddings = nn.Sequential(
            nn.Embedding(n_users, self.config.d_model * 3),
            nn.Linear(self.config.d_model * 3, self.config.d_model * 2),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(self.config.d_model * 2, self.config.d_model)
        )
        # self.relu = nn.LeakyReLU()

        self.post_init()

        self.to(device)

    def get_suggested_optimizer(self):
        return Adafactor(
            list(self.parameters()),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

    def set_eval_task(self, eval_task: Task):
        self.eval_task = eval_task

    def tokenize(self, sample):

        task = random.choice(self.training_tasks) if self.training else self.eval_task

        # give all info that we have about the sample to the task randomly sampled to generate
        # input prompt and target text. Each task may have mandatory arguments, if they are missing
        # an assertion error will be raised
        input_text, target_text = task(**sample)

        encoded_sequence = self.tokenizer(text=input_text, text_target=target_text, truncation=True)

        # get word ids from t5 tokenizer fast
        whole_word_ids = np.array(encoded_sequence.encodings[0].word_ids)
        special_token_mask = np.array(encoded_sequence.encodings[0].special_tokens_mask).astype(bool)

        # we set -1 to all special tokens (to substitute None, which is the value set by default)
        whole_word_ids[~special_token_mask] += 1
        whole_word_ids[special_token_mask] = self.tokenizer.pad_token_id

        encoded_sequence["user_idx"] = int(re.search(r"\d+", sample["user_id"]).group())
        encoded_sequence["whole_word_ids"] = whole_word_ids.tolist()
        encoded_sequence["target_item"] = sample["target_item"]

        return encoded_sequence

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
            input_dict["target_item"] = batch["target_item"]

        return input_dict

    def _inject_personalization(self, token_inputs_embeds: Tensor, user_idxs: Tensor):

        # whole_word_embeds = self.whole_word_embeddings(whole_word_ids)
        # # whole_word_embeds = self.relu(whole_word_embeds)
        # assert whole_word_embeds.shape[-1] == token_inputs_embeds.shape[-1]
        # inputs_embeds = token_inputs_embeds + whole_word_embeds

        # user idxs start from 1, TO IMPROVE!
        user_embeds = self.user_embeddings(user_idxs - 1).unsqueeze(axis=1)
        # whole_word_embeds = self.relu(whole_word_embeds)
        inputs_embeds = token_inputs_embeds + user_embeds

        return inputs_embeds

    def train_step(self, batch):

        inputs_embeds = self.shared(batch["input_ids"])  # embedding step - add HERE

        if "train" in ExperimentConfig.inject_personalization:
            inputs_embeds = self._inject_personalization(inputs_embeds, batch["user_idx"])

        output = self(inputs_embeds=inputs_embeds,
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"])

        return output.loss

    @torch.no_grad()
    def valid_step(self, batch):

        if self.eval_task is None:
            raise ValueError("Model can't perform valid_step since no eval_task is set! "
                             "Pass it when initializing the model or with `set_eval_task()`")

        num_return_sequences = 10
        max_new_tokens = 50
        num_beams = 30
        no_repeat_ngram_size = 0
        early_stopping = True

        target_text = batch.pop("target_item")

        inputs_embeds = self.shared(batch["input_ids"])
        if "eval" in ExperimentConfig.inject_personalization:
            inputs_embeds = self._inject_personalization(inputs_embeds, batch["user_idx"])

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
        # encoded_preds = sim_model.encode(generated_sents, show_progress_bar=False, convert_to_tensor=True)

        # sim = util.cos_sim(encoded_preds, self.encoded_all_labels).cpu()
        # mapped_predictions = self.all_unique_labels[sim.argmax(axis=1)]

        # mapped predictions is 1d. What we want is to have an array of shape (batch_size x num_return sequences)
        # mapped_predictions = mapped_predictions.reshape((len(target_text), num_return_sequences))

        mapped_predictions = np.array(generated_sents).reshape((len(target_text), num_return_sequences))
        val_loss = output.loss

        return mapped_predictions, target_text, val_loss
