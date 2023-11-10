from __future__ import annotations

import os.path
import random
from typing import List, Literal

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, Adafactor, T5TokenizerFast, AutoConfig

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
                 inject_personalization: bool = False,
                 eval_task_str: str = None,
                 eval_template_id: int | str = None,
                 train_task_selection_strat: Literal['random', 'all'] = "all",
                 **model_config_kwargs):

        super().__init__(
            name_or_path=name_or_path,
            training_tasks_str=training_tasks_str,
            all_unique_labels=all_unique_labels,
            eval_task_str=eval_task_str,
            eval_template_id=eval_template_id,
            train_task_selection_strat=train_task_selection_strat,
            **model_config_kwargs
        )

        self.model.config.inject_personalization = inject_personalization
        self.model.config.all_unique_users = all_unique_users

        self.user_embeddings = None
        self.user_mapping = {}

        if inject_personalization is True:
            if all_unique_users is None:
                raise AttributeError("all_unique_users parameter can't be None when "
                                     "inject_personalization is True!")

            for user in all_unique_users:
                self.user_mapping[user] = len(self.user_mapping)

            self.user_embeddings = UserEmbeds(len(all_unique_users), self.model.config.d_model).to(self.model.device)

    @property
    def get_suggested_optimizer(self):

        parameters = list(self.model.parameters())

        if self.user_embeddings is not None:
            parameters += list(self.user_embeddings.parameters())

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

                    encoded_sequence = self.tokenizer(text=input_text, text_target=target_text, truncation=True)

                    # get word ids from t5 tokenizer fast
                    whole_word_ids = np.array(encoded_sequence.encodings[0].word_ids)
                    special_token_mask = np.array(encoded_sequence.encodings[0].special_tokens_mask).astype(bool)

                    # we set -1 to all special tokens (to substitute None, which is the value set by default)
                    whole_word_ids[~special_token_mask] += 1
                    whole_word_ids[special_token_mask] = self.tokenizer.pad_token_id

                    # even if surely there is only one user, we wrap it into a list to be coherent
                    if self.model.config.inject_personalization is True:
                        encoded_sequence["user_idx"] = [self.user_mapping[sample["user_id"]]]

                    encoded_sequence["whole_word_ids"] = whole_word_ids.tolist()

                    if not self.model.training:

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

        input_dict["input_ids"] = input_ids.to(self.model.device)
        input_dict["attention_mask"] = attention_mask.to(self.model.device)
        input_dict["whole_word_ids"] = whole_word_ids.to(self.model.device)

        if "user_idx" in batch:
            # dim 1 is equal to 1, we don't need it since user_idxs will be passed
            # through the emb layer of UserEmbeds
            input_dict["user_idx"] = batch["user_idx"].to(self.model.device).squeeze()

        if "labels" in batch:
            lm_labels = pad_sequence(batch["labels"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

            input_dict["labels"] = lm_labels.to(self.model.device)

        if "gt" in batch:
            input_dict["gt"] = batch["gt"]

        return input_dict

    def _inject_personalization(self, token_inputs_embeds: Tensor, user_idxs: Tensor):

        # whole_word_embeds = self.whole_word_embeddings(whole_word_ids)
        # # whole_word_embeds = self.relu(whole_word_embeds)
        # assert whole_word_embeds.shape[-1] == token_inputs_embeds.shape[-1]
        # inputs_embeds = token_inputs_embeds + whole_word_embeds

        # unsqueeze to allow sum between token_inputs_embeds and user_embeds
        user_embeds = self.user_embeddings(user_idxs).unsqueeze(dim=1)
        inputs_embeds = token_inputs_embeds + user_embeds

        return inputs_embeds

    def train_step(self, batch):

        inputs_embeds = self.model.shared(batch["input_ids"])

        if self.model.config.inject_personalization is True:
            inputs_embeds = self._inject_personalization(inputs_embeds, batch["user_idx"])

        output = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"])

        return output.loss

    @torch.no_grad()
    def generate_step(self, batch, return_loss: bool = False):

        if self.eval_task is None:
            raise ValueError("Model can't perform generate_step since no eval_task is set! "
                             "Pass it when initializing the model or with `set_eval_task()`")

        if return_loss and "labels" not in batch:
            raise ValueError("Loss can't be returned if no label is set!")

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

        inputs_embeds = self.model.shared(batch["input_ids"])
        if self.model.config.inject_personalization is True:
            inputs_embeds = self._inject_personalization(inputs_embeds, batch["user_idx"])

        loss = torch.tensor(torch.nan)
        if return_loss is True:
            output = self.model(inputs_embeds=inputs_embeds,
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"])
            loss = output.loss

        beam_outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping
        )

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        mapped_predictions = np.array(generated_sents).reshape((len(gt), num_return_sequences))

        return mapped_predictions, gt, loss

    def to(self, device: str):
        if self.user_embeddings is not None:
            self.user_embeddings.to(device)

        return super().to(device)

    def save(self, output_dir: str):

        super().save(output_dir)

        if self.user_embeddings is not None:
            user_emb_out_pth = os.path.join(output_dir, "user_emb.pth")
            torch.save(self.user_embeddings.state_dict(), user_emb_out_pth)

    @classmethod
    def load(cls, dir_path: str, **config_and_laika_kwargs) -> T5Rec:

        # to avoid duplicate parameter error
        config_and_laika_kwargs.pop("return_unused_kwargs", None)

        config, laika_kwargs = AutoConfig.from_pretrained(dir_path,
                                                          **config_and_laika_kwargs,
                                                          return_unused_kwargs=True)

        # we can't pass **config, because model instantiation via config should be done
        # using .from_config() rather than .from_pretrained().
        # that's why we use config just to load parameters of LaikaModel serialized
        obj = cls(name_or_path=dir_path,
                  training_tasks_str=config.training_tasks_str,
                  all_unique_labels=config.all_unique_labels,
                  all_unique_users=config.all_unique_users,
                  inject_personalization=config.inject_personalization,
                  **laika_kwargs)

        if obj.user_embeddings is not None:
            user_emb_pth = os.path.join(dir_path, "user_emb.pth")
            obj.user_embeddings.load_state_dict(torch.load(user_emb_pth))

        return obj

    @classmethod
    def from_cls(cls, model_cls: type[T5Rec], dataset_obj: LaikaDataset, **kwargs):

        # n users is an additional requirement for t5 model that should
        # be extracted from dataset
        kwargs["all_unique_users"] = dataset_obj.all_users.tolist()

        return super().from_cls(model_cls, dataset_obj, **kwargs)
