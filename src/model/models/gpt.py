from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import List, Literal

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GenerationConfig, AutoConfig

from src.model.abstract_model import LaikaModelHF
from src.utils import dict_list2list_dict, list_dict2dict_list


class GPT2Rec(LaikaModelHF):
    model_class = GPT2LMHeadModel
    tokenizer_class = GPT2TokenizerFast

    def __init__(self,
                 name_or_path: str,
                 training_tasks_str: List[str],
                 all_unique_labels: List[str],
                 eval_task_str: str = None,
                 eval_template_id: int | str = None,
                 train_task_selection_strat: Literal['random', 'all'] = "all",
                 input_prefix: str = "Input: ",
                 target_prefix: str = "Target: ",
                 inject_whole_word_embeds: bool = True,
                 **model_config_and_gen_kwargs):

        # before passing the model config kwargs to super (which will pass them to the model config),
        # let's first consume the kwargs related to generation config
        # we set initial default values for relevant gen parameters that were not passed to __init__
        model_config_and_gen_kwargs["num_return_sequences"] = model_config_and_gen_kwargs.pop("num_return_sequences",
                                                                                              10)
        # max length is set using the model dim, we will set after super().__init() call
        model_config_and_gen_kwargs["max_length"] = model_config_and_gen_kwargs.pop("max_length", None)
        model_config_and_gen_kwargs["num_beams"] = model_config_and_gen_kwargs.pop("num_beams", 30)
        model_config_and_gen_kwargs["no_repeat_ngram_size"] = model_config_and_gen_kwargs.pop("no_repeat_ngram_size", 0)
        model_config_and_gen_kwargs["early_stopping"] = model_config_and_gen_kwargs.pop("early_stopping", True)

        generation_config, model_config_kwargs = GenerationConfig.from_pretrained(
            name_or_path, return_unused_kwargs=True, **model_config_and_gen_kwargs
        )

        super().__init__(
            name_or_path=name_or_path,
            training_tasks_str=training_tasks_str,
            all_unique_labels=all_unique_labels,
            eval_task_str=eval_task_str,
            eval_template_id=eval_template_id,
            train_task_selection_strat=train_task_selection_strat,
            **model_config_kwargs
        )

        self.model.generation_config = generation_config
        self.model.generation_config.max_length = self.tokenizer.model_max_length

        # gpt2 has no pad token, eos_token_id is used instead
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.model.config.input_prefix = input_prefix
        self.model.config.target_prefix = target_prefix
        self.model.config.inject_whole_word_embeds = inject_whole_word_embeds

        self.encoded_input_prefix = self.tokenizer(self.model.config.input_prefix,
                                                   return_attention_mask=False).input_ids
        self.encoded_target_prefix = self.tokenizer(self.model.config.target_prefix,
                                                    return_attention_mask=False).input_ids
        self.newline_token_id = self.tokenizer("\n", return_attention_mask=False).input_ids

        self.input_prefix_word_ids = None
        self.target_prefix_word_ids = None
        self.whole_word_embeddings = None
        if inject_whole_word_embeds is True:

            # By default, the tokenizer truncates to max 1024 tokens. At worst,
            # we expect 1024 different words (if custom max length is set to the tokenizer,
            # the first dimension should be changed here too)
            self.whole_word_embeddings = nn.Embedding(
                self.tokenizer.model_max_length, self.model.config.hidden_size
            ).to(self.model.device)

            self.input_prefix_word_ids = np.array(self.tokenizer(self.model.config.input_prefix).word_ids(0))
            self.target_prefix_word_ids = np.array(self.tokenizer(self.model.config.target_prefix).word_ids(0))

    @property
    def get_suggested_optimizer(self):

        parameters = list(self.model.parameters())

        if self.whole_word_embeddings is not None:
            parameters += list(self.whole_word_embeddings.parameters())

        return AdamW(
            parameters,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

    def tokenize(self, batch):

        if "user_id" not in batch:
            raise AttributeError("This model expects 'user_id' column in the dataset to tokenize!")

        if not self.model.training and self.eval_task is None:
            raise ValueError("Model can't tokenize the eval task since no eval_task is set! "
                             "Pass it when initializing the model or with `set_eval_task()`")

        # from dict of lists to list of dicts
        batch = dict_list2list_dict(batch)

        encoded_sequence_list = []
        for sample in batch:

            if not self.model.training:
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

                    # <end of text token> so we enforce the fact that the model should make the prediction
                    # (represented by the target text) and that's it! No endless generation!
                    target_encoded_sequence = self.tokenizer(f"{target_text}<|endoftext|>",
                                                             truncation=True,
                                                             return_attention_mask=False)
                    target_ids = target_encoded_sequence.input_ids

                    len_reserved_target = len(self.newline_token_id) + len(self.encoded_target_prefix) + len(target_ids)

                    input_text_encoded_sequence = self.tokenizer(
                        f"{self.model.config.input_prefix}{input_text} ",
                        truncation=True,
                        max_length=self.tokenizer.model_max_length - len_reserved_target,
                        return_attention_mask=False)
                    input_text_ids = input_text_encoded_sequence.input_ids

                    # why we add later newline, target_prefix and target? due to POSSIBLE TRUNCATION!
                    # in this way, the context may be truncated, but the target WILL BE NOT
                    input_text_ids += self.newline_token_id + self.encoded_target_prefix

                    total_input_ids = input_text_ids + target_ids

                    encoded_sequence: dict = {
                        "input_prompt_ids": input_text_ids,
                        "input_prompt_attention_mask": [1] * len(input_text_ids),
                        "total_input_ids": total_input_ids,
                        "total_attention_mask": [1] * len(total_input_ids),

                        # objective is to reconstruct the input text + target
                        "total_labels": deepcopy(total_input_ids)
                    }

                    if self.model.config.inject_whole_word_embeds is True:
                        input_whole_word_ids, total_whole_word_ids = self._tokenize_whole_word_ids(
                            input_text_encoded_sequence.word_ids(0),
                            target_encoded_sequence.word_ids(0)
                        )

                        assert len(input_whole_word_ids) == len(input_text_ids)
                        assert len(total_whole_word_ids) == len(total_input_ids)

                        encoded_sequence["input_whole_word_ids"] = input_whole_word_ids.tolist()
                        encoded_sequence["total_whole_word_ids"] = total_whole_word_ids.tolist()

                    if not self.model.training:
                        if gt is None:
                            raise ValueError("In the __call__ method of the template, the `gt` attribute should be "
                                             "set for templates used in the evaluation phase!")

                        # it may be the item id or the item rating for example, depending on the task chosen
                        encoded_sequence["gt"] = gt

                    encoded_sequence_list.append(encoded_sequence)

        # from list of dicts to dict of lists
        return list_dict2dict_list(encoded_sequence_list)

    def prepare_input(self, batch: dict):
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

        input_dict["total_input_ids"] = total_input_ids.to(self.model.device)
        input_dict["total_attention_mask"] = total_attention_mask.to(self.model.device)

        input_dict["input_prompt_ids"] = input_prompt_ids.to(self.model.device)
        input_dict["input_prompt_attention_mask"] = input_prompt_attention_mask.to(self.model.device)
        input_dict["total_labels"] = lm_labels.to(self.model.device)

        if self.model.config.inject_whole_word_embeds is True:
            input_whole_word_ids = pad_sequence(
                batch["input_whole_word_ids"],
                batch_first=True,
                padding_value=0
            )
            total_whole_word_ids = pad_sequence(
                batch["total_whole_word_ids"],
                batch_first=True,
                padding_value=0
            )

            input_dict["input_whole_word_ids"] = input_whole_word_ids.to(self.model.device)
            input_dict["total_whole_word_ids"] = total_whole_word_ids.to(self.model.device)

        if "gt" in batch:
            input_dict["gt"] = batch["gt"]

        return input_dict

    def _tokenize_whole_word_ids(self, input_whole_word_ids: list, target_whole_word_ids: list):

        # get word ids from gpt2 tokenizer fast
        # gpt2 tokenizer doesn't add any special token, so this snipped is under the assumption
        # that no special token is added by the tokenizer
        # (and so word_ids arrays do not have any None value)
        input_whole_word_ids.append(input_whole_word_ids[-1] + 1)  # newline token
        input_whole_word_ids = np.array(input_whole_word_ids)

        # adding the "Target: " prefix, but the word ids count should continue where the
        # "input_whole_word_ids" left
        offset_target_prefix_word_ids = self.target_prefix_word_ids + input_whole_word_ids[-1] + 1
        input_whole_word_ids = np.concatenate((input_whole_word_ids, offset_target_prefix_word_ids))

        # adding the target sequence, that is used in the training phase only.
        # again, word ids count should continue from where we left off
        offset_target_word_ids = np.array(target_whole_word_ids) + input_whole_word_ids[-1] + 1
        total_whole_word_ids = np.concatenate((input_whole_word_ids,
                                               offset_target_word_ids))

        # increase word ids count starting from 1 rather than 0
        input_whole_word_ids += 1
        total_whole_word_ids += 1

        # gpt2 tokenizer doesn't add any special token, but the last token is
        # surely the eos token due to how we set the target sequence to tokenize.
        if len(total_whole_word_ids) != 0:
            total_whole_word_ids[-1] = 0

        return input_whole_word_ids, total_whole_word_ids

    def _inject_whole_word_embeds(self, token_inputs_embeds: Tensor, whole_word_ids: Tensor):

        whole_word_embeds = self.whole_word_embeddings(whole_word_ids)
        assert whole_word_embeds.shape[-1] == token_inputs_embeds.shape[-1]
        inputs_embeds = token_inputs_embeds + whole_word_embeds

        return inputs_embeds

    def train_step(self, batch: dict):

        inputs_embeds = self.model.transformer.wte(batch["total_input_ids"])

        if self.model.config.inject_whole_word_embeds is True:
            inputs_embeds = self._inject_whole_word_embeds(inputs_embeds, batch["total_whole_word_ids"])

        output = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=batch["total_attention_mask"],
                            labels=batch["total_labels"])

        return output.loss

    @torch.no_grad()
    def generate_step(self, batch: dict, return_loss: bool = False):

        if self.eval_task is None:
            raise ValueError("Model can't perform generate_step since no eval_task is set! "
                             "Pass it when initializing the model or with `set_eval_task()`")

        if return_loss and "labels" not in batch:
            raise ValueError("Loss can't be returned if no label is set!")

        # if it's not a ranking task (e.g., it is a rating prediction task),
        # we should return one prediction for ground truth element.
        # In theory there could be a complex way of mixing multiple text generated into a single prediction
        # (e.g., avg of 10 rating predictions), but here we simply reduce the num return sequences
        num_return_sequences = self.model.generation_config.num_return_sequences
        if not self.eval_task.is_ranking_task():
            num_return_sequences = 1

        gt = np.array(batch.pop("gt"))

        loss = torch.tensor(torch.nan)
        if return_loss is True:
            # this does not update gradients since we use decorator torch.no_grad()
            loss = self.train_step(batch)

        # for decoder only model, input should be padded to the left when performing batch inference with generate,
        # otherwise you are continuing generating over a pad token which was little meaning!
        # Check https://github.com/huggingface/transformers/issues/3021#issuecomment-1454267951
        left_padded_input_ids = self._left_pad(batch["input_prompt_ids"],
                                               pad_token=self.tokenizer.pad_token_id)
        left_padded_attn_mask = self._left_pad(batch["input_prompt_attention_mask"],
                                               pad_token=0)

        inputs_embeds = self.model.transformer.wte(left_padded_input_ids)

        if self.model.config.inject_whole_word_embeds is True:
            left_padded_word_ids = self._left_pad(batch["input_whole_word_ids"], pad_token=0)
            inputs_embeds = self._inject_whole_word_embeds(inputs_embeds, left_padded_word_ids)

        # for some decoder only models (in particular gpt2) it is possible to perform generate using
        # custom inputs_embeds. They are used at the 1st step of the generation process only.
        # It is needed to pass also "input_ids" so that the input prompt is returned in output
        beam_outputs = self.model.generate(
            input_ids=left_padded_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=left_padded_attn_mask,
            num_return_sequences=num_return_sequences,
            generation_config=self.model.generation_config
        )

        # this works for all rows of tensor because, when generating, also pad tokens are generated,
        # so the index where the target prediction starts is in common to all rows!
        beam_outputs_targets = beam_outputs[:, inputs_embeds.shape[1]:]

        generated_sents = self.tokenizer.batch_decode(beam_outputs_targets, skip_special_tokens=True)
        mapped_predictions = np.array(generated_sents).reshape((len(gt), num_return_sequences))

        return mapped_predictions, gt, loss

    def _left_pad(self, right_padded_tensor: torch.Tensor, pad_token: int):

        # calculate the number of padding tokens in each row, which is where
        # the valid ids start for each row
        num_padding_tokens = (right_padded_tensor == pad_token).sum(dim=1)

        # create a new tensor filled with padding tokens (zero tokens for attention mask)
        left_padded_tensor = torch.full_like(right_padded_tensor, fill_value=pad_token)

        # For each row, move the actual content to the specified positions
        for i in range(right_padded_tensor.size(0)):
            end_index_valid_ids = right_padded_tensor.size(1) - num_padding_tokens[i]
            left_padded_tensor[i, num_padding_tokens[i]:] = right_padded_tensor[i, :end_index_valid_ids]

        return left_padded_tensor

    @torch.no_grad()
    def inference(self, input_text: str | list[str], format_input: bool = True, return_only_target: bool = False,
                  **gen_config):

        if not isinstance(input_text, list):
            input_text = [input_text]

        if format_input is True:
            input_text = [f"{self.model.config.input_prefix}{inp} \n{self.model.config.target_prefix}"
                          for inp in input_text]

        encoded_inputs = self.tokenizer(input_text,
                                        truncation=True,
                                        padding=True,
                                        return_tensors="pt")

        left_padded_input_ids = self._left_pad(encoded_inputs.input_ids,
                                               pad_token=self.tokenizer.pad_token_id).to(self.model.device)
        left_padded_attn_mask = self._left_pad(encoded_inputs.attention_mask,
                                               pad_token=0).to(self.model.device)

        inputs_embeds = self.model.transformer.wte(left_padded_input_ids)

        if self.model.config.inject_whole_word_embeds is True:
            whole_word_ids = np.array([encoded_inputs.word_ids(i) for i in range(len(input_text))])

            special_tokens_mask = whole_word_ids == None

            whole_word_ids[~special_tokens_mask] += 1
            whole_word_ids[special_tokens_mask] = 0

            whole_word_ids = torch.tensor(whole_word_ids.astype(int)).to(self.model.device)

            left_padded_word_ids = self._left_pad(whole_word_ids, pad_token=0)
            inputs_embeds = self._inject_whole_word_embeds(inputs_embeds, left_padded_word_ids)

        beam_outputs = self.model.generate(
            input_ids=left_padded_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=left_padded_attn_mask,
            generation_config=self.model.generation_config,
            **gen_config
        )

        if return_only_target is True:
            beam_outputs = beam_outputs[:, left_padded_input_ids.shape[1]:]

        num_return_sequences = gen_config.get("num_return_sequences", self.model.generation_config.num_return_sequences)

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        mapped_predictions = np.array(generated_sents).reshape((len(input_text),
                                                                num_return_sequences))

        return mapped_predictions.tolist()

    def to(self, device: str):
        if self.whole_word_embeddings is not None:
            self.whole_word_embeddings.to(device)

        return super().to(device)

    def save(self, output_dir: str):

        super().save(output_dir)

        if self.whole_word_embeddings is not None:
            whole_word_emb_out_pth = os.path.join(output_dir, "whole_word_emb.pth")
            torch.save(self.whole_word_embeddings.state_dict(), whole_word_emb_out_pth)

    @classmethod
    def load(cls, dir_path: str, **config_gen_laika_kwargs) -> GPT2Rec:

        gen_config, config_laika_kwargs = GenerationConfig.from_pretrained(dir_path,
                                                                           **config_gen_laika_kwargs,
                                                                           return_unused_kwargs=True)

        config, laika_kwargs = AutoConfig.from_pretrained(dir_path,
                                                          **config_laika_kwargs,
                                                          return_unused_kwargs=True)

        # all parameters were basically saved inside the model config and are loaded back
        # automatically, but we need to pass `inject_whole_word_embeds`
        # so that they are initialized in case they are needed. Their state dicts is loaded
        # below
        obj: GPT2Rec = cls(name_or_path=dir_path,
                           training_tasks_str=config.training_tasks_str,
                           all_unique_labels=config.all_unique_labels,
                           inject_whole_word_embeds=config.inject_whole_word_embeds,

                           **laika_kwargs)

        obj.model.config = config
        obj.model.generation_config = gen_config

        # we load back the whole word emb layer weights if needed
        if obj.whole_word_embeddings is not None:
            whole_word_emb_pth = os.path.join(dir_path, "whole_word_emb.pth")
            obj.whole_word_embeddings.load_state_dict(torch.load(whole_word_emb_pth))

        return obj
