from __future__ import annotations

import inspect
from abc import abstractmethod, ABC
from typing import List, Optional, Literal

import numpy as np
import torch
from requests.structures import CaseInsensitiveDict
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig, AutoTokenizer

from src.data.abstract_dataset import LaikaDataset
from src.data.abstract_task import LaikaTask


class LaikaModel(ABC):
    str_alias_cls: dict[str, type[LaikaModel]] = CaseInsensitiveDict()

    # automatically called on subclass definition, will populate the str_alias_cls dict
    def __init_subclass__(cls, **kwargs):
        if not inspect.isabstract(cls):
            cls.str_alias_cls[cls.__name__] = cls

        super().__init_subclass__(**kwargs)

    def __init__(self, training_tasks_str: List[str],
                 all_unique_labels: List[str],
                 eval_task_str: str = None,
                 eval_template_id: int | str = None,
                 train_task_selection_strat: Literal['random', 'all'] = "all"):

        if training_tasks_str is None:
            raise AttributeError("training_tasks_str parameter can't be None!")
        if all_unique_labels is None:
            raise AttributeError("all_unique_labels parameter can't be None!")
        if train_task_selection_strat not in {"random", "all"}:
            raise AttributeError("train_task_selection_strat should be 'all' or 'random'!")

        self.all_unique_labels = np.array(all_unique_labels)
        self.training_tasks = [LaikaTask.from_string(training_task_str) for training_task_str in training_tasks_str]

        self.eval_task: Optional[LaikaTask] = None
        if eval_task_str is not None:
            self.set_eval_task(eval_task_str, eval_template_id)

        self.train_task_selection_strat = train_task_selection_strat

    def set_eval_task(self, eval_task_str: str, template_id: int = None):
        self.eval_task = LaikaTask.from_string(eval_task_str)

        if template_id is not None:
            self.eval_task.force_template(template_id)

    @property
    @abstractmethod
    def get_suggested_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, batch: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def prepare_input(self, tokenized_batch: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, prepared_batch: dict) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def generate_step(self, prepared_batch: dict, return_loss: bool = False) -> tuple[np.ndarray[np.ndarray[str]],
                                                                                      np.ndarray[np.ndarray[str]],
                                                                                      torch.FloatTensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def inference(self, input_text: str | list[str], **kwargs) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def train(self, mode: bool = True):
        raise NotImplementedError

    def eval(self):
        LaikaTask.eval()

        self.train(False)

    @abstractmethod
    def save(self, output_dir: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str, **kwargs) -> LaikaModel:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: str):
        raise NotImplementedError

    @classmethod
    def from_cls(cls, model_cls: type[LaikaModel], dataset_obj: LaikaDataset, **kwargs) -> LaikaModel:
        raise NotImplementedError

    @classmethod
    def from_string(cls, model_cls_name: str, dataset_obj: LaikaDataset, **kwargs) -> LaikaModel:

        model_cls = cls.model_exists(model_cls_name, return_bool=False)

        # it seems a recursive call, but the top level (LaikaModel) is an abstract class,
        # model_cls is a concrete class
        return model_cls.from_cls(model_cls, dataset_obj, **kwargs)

    @classmethod
    def all_models_available(cls, return_str: bool = False) -> list[type[LaikaModel] | str]:
        return list(cls.str_alias_cls.keys()) if return_str else list(cls.str_alias_cls.values())

    @classmethod
    def model_exists(cls, model_cls_name: str, return_bool: bool = True) -> bool | type[LaikaModel]:

        try:
            model_cls = cls.str_alias_cls[model_cls_name]
        except KeyError:
            raise KeyError(f"Dataset {model_cls_name} does not exist!") from None

        # if we arrive at the return clause, model_cls exists that's why we return True directly
        return model_cls if not return_bool else True


# this is for pretrained hf model. Maybe in the future an alternative class can be
# made where we call the init of the hf model rather than 'from_pretrained()'
class LaikaModelHF(LaikaModel):
    # model class is mandatory, since same model family
    # exist ModelForCausalLM, ModelForConditionalGeneration, etc.
    model_class: type[PreTrainedModel] = None

    # if tokenizer class is not specified by the subclass, AutoTokenizer will be used
    tokenizer_class: type[PreTrainedTokenizer] = AutoTokenizer

    def __init__(self,
                 name_or_path: str,
                 training_tasks_str: List[str],
                 all_unique_labels: List[str],
                 eval_task_str: str = None,
                 eval_template_id: int | str = None,
                 train_task_selection_strat: Literal['random', 'all'] = "all",
                 **model_config_kwargs):

        super().__init__(training_tasks_str=training_tasks_str,
                         all_unique_labels=all_unique_labels,
                         eval_task_str=eval_task_str,
                         eval_template_id=eval_template_id,
                         train_task_selection_strat=train_task_selection_strat)

        if self.model_class is None:
            raise AttributeError("Please set the class attribute 'model_class' when extending LaikaModelHF!")

        self.model = self.model_class.from_pretrained(name_or_path, **model_config_kwargs)
        self.tokenizer = self.tokenizer_class.from_pretrained(name_or_path)

        # store in model config all parameters needed to re-instantiate the hf model,
        # so to exploit serialization and de-serialization of hf with from_pretrained()
        self.model.config.training_tasks_str = training_tasks_str
        self.model.config.all_unique_labels = all_unique_labels

    def train(self, mode: bool = True):

        if mode is True:
            LaikaTask.train()
        else:
            LaikaTask.eval()

        self.model.train(mode=mode)

    def save(self, output_dir: str):
        # save hf model and parameters that we added to the config
        self.model.save_pretrained(save_directory=output_dir)

        # also tokenizer is saved
        self.tokenizer.save_pretrained(save_directory=output_dir)

    @classmethod
    # this method should be subclassed whenever the model has any additional parameter
    # that is NOT stored inside the hugging face model config
    def load(cls, dir_path: str, **config_and_laika_kwargs) -> LaikaModelHF:

        # laika kwargs for example are val_template, val_template_id, etc.
        config, laika_kwargs = AutoConfig.from_pretrained(dir_path,
                                                          **config_and_laika_kwargs,
                                                          return_unused_kwargs=True)

        # we use config to load mandatory parameters of LaikaModel serialized
        # in this case **laika_kwargs are those not saved into the model config
        obj = cls(name_or_path=dir_path,
                  training_tasks_str=config.training_tasks_str,
                  all_unique_labels=config.all_unique_labels,
                  **laika_kwargs)

        # regardless of what happens in init, we will substitute the initialized
        # config with the loaded config, to avoid re-initialization to possible default
        # values since we passed through __init__ again.
        # NOTE: this loaded config already has updated laika kwargs, if they were saved into the config
        # and new values are passed to this function through **kwargs
        obj.model.config = config

        return obj

    def to(self, device: str):
        return self.model.to(device)

    @classmethod
    def from_cls(cls, model_cls: type[LaikaModelHF], dataset_obj: LaikaDataset, **kwargs) -> LaikaModelHF:

        kwargs["all_unique_labels"] = dataset_obj.all_items.tolist()

        return model_cls(**kwargs)
