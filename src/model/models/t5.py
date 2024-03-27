from __future__ import annotations

import os.path
import random
from typing import List, Literal

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, Adafactor, T5TokenizerFast, AutoConfig, GenerationConfig

from src.data.abstract_dataset import LaikaDataset
from src.model.abstract_model import LaikaModelHF
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


class T5Rec(LaikaModelHF):
    model_class = T5ForConditionalGeneration
    tokenizer_class = T5TokenizerFast

    def __init__(self,
                 name_or_path: str,
                 training_tasks_str: List[str],
                 all_unique_labels: List[str],
                 all_unique_users: List[str] = None,
                 inject_user_embeds: bool = False,
                 inject_whole_word_embeds: bool = False,
                 eval_task_str: str = None,
                 eval_template_id: int | str = None,
                 train_task_selection_strat: Literal['random', 'all'] = "all",
                 **model_config_and_gen_kwargs):

        # before passing the model config kwargs to super (which will pass them to the model config),
        # let's first consume the kwargs related to generation config
        # we set initial default values for relevant gen parameters that were not passed to __init__
        model_config_and_gen_kwargs["num_return_sequences"] = model_config_and_gen_kwargs.pop("num_return_sequences",
                                                                                              10)
        model_config_and_gen_kwargs["max_new_tokens"] = model_config_and_gen_kwargs.pop("max_new_tokens", 50)
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

        self.model.config.inject_user_embeds = inject_user_embeds
        self.model.config.inject_whole_word_embeds = inject_whole_word_embeds
        self.model.config.all_unique_users = all_unique_users

        self.model.config.user_mapping = {}
        self.user_embeddings = None
        if inject_user_embeds is True:
            if all_unique_users is None:
                raise AttributeError("all_unique_users parameter can't be None when "
                                     "inject_user_embeds is True!")

            for user in all_unique_users:
                self.model.config.user_mapping[user] = len(self.model.config.user_mapping)

            self.user_embeddings = UserEmbeds(len(all_unique_users),
                                              self.model.config.d_model).to(self.model.device)

        self.whole_word_embeddings = None
        if inject_whole_word_embeds is True:
            self.whole_word_embeddings = nn.Embedding(
                512, self.model.config.d_model  # config.d_model is 768 for base
            )

    @property
    def get_suggested_optimizer(self):

        parameters = list(self.model.parameters())

        if self.user_embeddings is not None:
            parameters += list(self.user_embeddings.parameters())

        if self.whole_word_embeddings is not None:
            parameters += list(self.whole_word_embeddings.parameters())

        return Adafactor(
            parameters,
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

    def tokenize(self, batch: dict):

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

                    encoded_sequence = self.tokenizer(text=input_text, text_target=target_text, truncation=True)

                    if self.model.config.inject_whole_word_embeds is True:
                        # get word ids from t5 tokenizer fast
                        whole_word_ids = np.array(encoded_sequence.encodings[0].word_ids)
                        special_token_mask = np.array(encoded_sequence.encodings[0].special_tokens_mask).astype(bool)

                        # we set -1 to all special tokens (to substitute None, which is the value set by default)
                        whole_word_ids[~special_token_mask] += 1
                        whole_word_ids[special_token_mask] = self.tokenizer.pad_token_id

                        encoded_sequence["whole_word_ids"] = whole_word_ids.tolist()

                    # even if surely there is only one user, we wrap it into a list to be coherent
                    if self.model.config.inject_user_embeds is True:
                        encoded_sequence["user_idx"] = [self.model.config.user_mapping[sample["user_id"]]]

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

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"],
                                      batch_first=True,
                                      padding_value=0)

        input_dict["input_ids"] = input_ids.to(self.model.device)
        input_dict["attention_mask"] = attention_mask.to(self.model.device)

        if "whole_word_ids" in batch:
            whole_word_ids = pad_sequence(batch["whole_word_ids"],
                                          batch_first=True,
                                          padding_value=0)
            input_dict["whole_word_ids"] = whole_word_ids.to(self.model.device)

        if "user_idx" in batch:
            # dim 1 is equal to 1, we don't need it since user_idxs will be passed
            # through the emb layer of UserEmbeds
            input_dict["user_idx"] = batch["user_idx"].to(self.model.device).squeeze()

        if "labels" in batch:
            lm_labels = pad_sequence(batch["labels"], batch_first=True, padding_value=-100)
            input_dict["labels"] = lm_labels.to(self.model.device)

        if "gt" in batch:
            input_dict["gt"] = batch["gt"]

        return input_dict

    def _inject_whole_word_embeds(self, token_inputs_embeds: Tensor, whole_word_ids: Tensor):

        whole_word_embeds = self.whole_word_embeddings(whole_word_ids)
        assert whole_word_embeds.shape[-1] == token_inputs_embeds.shape[-1]
        inputs_embeds = token_inputs_embeds + whole_word_embeds

        return inputs_embeds

    def _inject_user_embeds(self, token_inputs_embeds: Tensor, user_idxs: Tensor):

        # unsqueeze to allow sum between token_inputs_embeds and user_embeds
        user_embeds = self.user_embeddings(user_idxs).unsqueeze(dim=1)
        inputs_embeds = token_inputs_embeds + user_embeds

        return inputs_embeds

    def train_step(self, batch: dict):

        inputs_embeds = self.model.shared(batch["input_ids"])

        if self.model.config.inject_user_embeds is True:
            inputs_embeds = self._inject_user_embeds(inputs_embeds, batch["user_idx"])

        if self.model.config.inject_whole_word_embeds is True:
            inputs_embeds = self._inject_whole_word_embeds(inputs_embeds, batch["whole_word_ids"])

        output = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"])

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

        inputs_embeds = self.model.shared(batch["input_ids"])
        if self.model.config.inject_user_embeds is True:
            inputs_embeds = self._inject_user_embeds(inputs_embeds, batch["user_idx"])

        if self.model.config.inject_whole_word_embeds is True:
            inputs_embeds = self._inject_whole_word_embeds(inputs_embeds, batch["whole_word_ids"])

        loss = torch.tensor(torch.nan)
        if return_loss is True:
            output = self.model(inputs_embeds=inputs_embeds,
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"])
            loss = output.loss

        beam_outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            generation_config=self.model.generation_config,
            num_return_sequences=num_return_sequences
        )

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        mapped_predictions = np.array(generated_sents).reshape((len(gt), num_return_sequences))

        return mapped_predictions, gt, loss

    @torch.no_grad()
    def inference(self, input_text: str | list[str], user_id: str | list[str] = None, **gen_config):
        # if inject_user_embeds is True, `input_text` and `user_id` should be in 1:1 relationship,
        # meaning that each input text should be related to a particular user already known to the user

        if not isinstance(input_text, list):
            input_text = [input_text]

        if not isinstance(user_id, list):
            user_id = [user_id]

        encoded_inputs = self.tokenizer(input_text,
                                        truncation=True,
                                        padding=True,
                                        return_tensors="pt")

        input_ids = encoded_inputs.input_ids.to(self.model.device)
        attention_mask = encoded_inputs.attention_mask.to(self.model.device)

        inputs_embeds = self.model.shared(input_ids)
        if self.model.config.inject_user_embeds is True:

            # we are sure there is an element since we did the wrapping
            if user_id[0] is None:
                raise ValueError("Model was fine-tuned with `inject_user_embeds`, please for each input text "
                                 "specify to which user it refers to with the `user_id` parameter")

            elif len(input_text) != len(user_id):
                raise ValueError("Each input text should be related to a known user, so `input_text` parameter and "
                                 "`user_id` parameter should be in 1:1 relationship!")

            try:
                mapped_user_ids = [self.model.config.user_mapping[user] for user in user_id]
                mapped_user_ids = torch.tensor(mapped_user_ids).to(self.model.device)

                inputs_embeds = self._inject_user_embeds(inputs_embeds, mapped_user_ids)
            except KeyError as e:
                missing_key = e.args[0]
                raise KeyError(f"User {missing_key} was not known at train time!") from None

        if self.model.config.inject_whole_word_embeds is True:

            # get word ids from t5 tokenizer fast
            whole_word_ids = np.array([encoded_inputs.encodings[i].word_ids for i in range(len(input_text))])
            special_token_mask = np.array([encoded_inputs.encodings[i].special_tokens_mask
                                          for i in range(len(input_text))]).astype(bool)

            # we set -1 to all special tokens (to substitute None, which is the value set by default)
            whole_word_ids[~special_token_mask] += 1
            whole_word_ids[special_token_mask] = self.tokenizer.pad_token_id

            whole_word_ids = torch.tensor(whole_word_ids.astype(int)).to(self.model.device)

            inputs_embeds = self._inject_whole_word_embeds(inputs_embeds, whole_word_ids)

        beam_outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=self.model.generation_config,
            **gen_config
        )

        num_return_sequences = gen_config.get("num_return_sequences", self.model.generation_config.num_return_sequences)

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        mapped_predictions = np.array(generated_sents).reshape((len(input_text),
                                                                num_return_sequences))

        return mapped_predictions.tolist()

    def to(self, device: str):
        if self.user_embeddings is not None:
            self.user_embeddings.to(device)

        if self.whole_word_embeddings is not None:
            self.whole_word_embeddings.to(device)

        return super().to(device)

    def save(self, output_dir: str):

        super().save(output_dir)

        if self.user_embeddings is not None:
            user_emb_out_pth = os.path.join(output_dir, "user_emb.pth")
            torch.save(self.user_embeddings.state_dict(), user_emb_out_pth)

        if self.whole_word_embeddings is not None:
            whole_word_emb_out_pth = os.path.join(output_dir, "whole_word_emb.pth")
            torch.save(self.whole_word_embeddings.state_dict(), whole_word_emb_out_pth)

    @classmethod
    def load(cls, dir_path: str, **config_gen_laika_kwargs) -> T5Rec:

        gen_config, config_laika_kwargs = GenerationConfig.from_pretrained(dir_path,
                                                                           **config_gen_laika_kwargs,
                                                                           return_unused_kwargs=True)

        config, laika_kwargs = AutoConfig.from_pretrained(dir_path,
                                                          **config_laika_kwargs,
                                                          return_unused_kwargs=True)

        # all parameters were basically saved inside the model config and are loaded back
        # automatically, but (apart from the mandatory parameters) we need to pass
        # `inject_user_embeds` and `inject_whole_word_embeds`
        # so that they are initialized in case they are needed. Their state dicts is loaded below
        obj = cls(name_or_path=dir_path,
                  training_tasks_str=config.training_tasks_str,
                  all_unique_labels=config.all_unique_labels,
                  all_unique_users=config.all_unique_users,
                  inject_user_embeds=config.inject_user_embeds,
                  inject_whole_word_embeds=config.inject_whole_word_embeds,

                  **laika_kwargs)

        obj.model.config = config
        obj.model.generation_config = gen_config

        # we load back additional weights back if needed
        if obj.user_embeddings is not None:
            user_emb_pth = os.path.join(dir_path, "user_emb.pth")
            obj.user_embeddings.load_state_dict(torch.load(user_emb_pth))

        if obj.whole_word_embeddings is not None:
            whole_word_emb_pth = os.path.join(dir_path, "whole_word_emb.pth")
            obj.whole_word_embeddings.load_state_dict(torch.load(whole_word_emb_pth))

        return obj

    @classmethod
    def from_cls(cls, model_cls: type[T5Rec], dataset_obj: LaikaDataset, **kwargs):

        # all_unique_users is an additional requirement for t5 model that should
        # be extracted from dataset
        kwargs["all_unique_users"] = dataset_obj.all_users.tolist()

        return super().from_cls(model_cls, dataset_obj, **kwargs)
