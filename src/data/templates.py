import itertools
from abc import ABC, abstractmethod
import random

import numpy as np
from requests.structures import CaseInsensitiveDict


class PromptTarget:

    def __init__(self, input_prompt: str, target_text: str):
        self.input_prompt = input_prompt
        self.target_text = target_text

    # iter just so that this class can be unpacked,
    # e.g. input_prompt, target_text = PromptTarget(...)
    def __iter__(self):
        return iter((self.input_prompt, self.target_text))

    def __str__(self):
        string = " Input ".center(50, "#") + "\n"
        string += self.input_prompt + "\n"
        string += " Target ".center(50, "#") + "\n"
        string += self.target_text

        return string


class Task(ABC):
    # keys are integers, values are PromptTarget objects
    templates_dict = {}
    # name obj class mapping, used for when task must be initialized from strings
    str_alias_obj: dict = CaseInsensitiveDict()
    # class attribute since if the model is in training mode, all tasks should be in training mode
    training = False

    def __init__(self, all_unique_items: np.ndarray[str]):
        self.all_unique_items = all_unique_items

    # automatically called on subclass definition, will populate the str_alias_obj dict
    def __init_subclass__(cls, **kwargs):
        cls.str_alias_obj[cls.__name__] = cls

    @classmethod
    def train(cls):
        Task.training = True

    @classmethod
    def eval(cls):
        Task.training = False

    def all_templates(self, return_id: bool = False):
        return list(self.templates_dict.keys()) if return_id else list(self.templates_dict.values())

    @abstractmethod
    def valid_templates(self, return_id: bool = False):
        raise NotImplementedError

    @abstractmethod
    def support_templates(self, return_id: bool = False):
        raise NotImplementedError

    def force_template(self, force_template_id: int):

        # 'self.__class__' so that even if we call multiple times this method on an instantiated task,
        # we always have a pointer to original class templates, otherwise they are deleted if we use simply 'self'
        # instead of self.__class__

        if force_template_id not in set(self.__class__.templates_dict.keys()):
            raise KeyError(f"Prompt template id {force_template_id} not found! "
                           f"Available prompt ids are {list(self.templates_dict.keys())}")

        self.templates_dict = {force_template_id: self.__class__.templates_dict[force_template_id]}

        return self

    # function decorator needed to declare mandatory arguments of each subclass __call__
    @staticmethod
    def validate_args(*mandatory_args: str):
        def decorator(func):
            def wrapper(self, **kwargs):
                for mandatory_arg in mandatory_args:
                    assert mandatory_arg in kwargs, f"{mandatory_arg} is needed for task {repr(self)}!"

                return func(self, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def from_string(cls, *task_str: str, all_unique_items: np.ndarray[str]):

        try:
            # remember, we are searching a case-insensitive dict, so we don't care about
            # lowering all keys
            instantiated_task = [cls.str_alias_obj[task](all_unique_items) for task in task_str]
        except KeyError:
            raise KeyError("One or more task string alias does not exist!") from None

        return instantiated_task

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return repr(self)


class SequentialTask(Task):
    templates_dict = {
        0: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "Predict for the user the next element of the following sequence -> \n"
                         "{}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "Predict the next element which the user will buy given the following order history -> \n"
                         "{}",
            target_text="{}"
        ),
        2: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "What is the element that should be recommended to the user knowing that it has bought -> \n"
                         "{}",
            target_text="{}"
        ),
        3: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "Recommend to the user an item from the catalog given its order history -> \n"
                         "{}",
            target_text="{}"
        ),
        4: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "This is the order history of the user -> \n"
                         "{} \n"
                         "Recommend the next element that the user will buy",
            target_text="{}"
        ),
        5: PromptTarget(
            input_prompt="sequential_rec for {}: \n\n"
                         "Please predict what item is best to recommend to the user given its order history -> \n"
                         "{}",
            target_text="{}"
        )
    }

    def valid_templates(self, return_id: bool = True):
        return self.all_templates(return_id)

    def support_templates(self, return_id: bool = True):
        return []

    @Task.validate_args("user_id", "input_item_seq", "target_item")
    def __call__(self, **kwargs):
        user_id = kwargs["user_id"]
        order_history = kwargs["input_item_seq"]
        target_item = kwargs["target_item"]

        # random.choice applied to dict with int key returns a value
        input_prompt, target = random.choice(self.all_templates())

        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        order_history_str = separator.join(order_history)

        input_text = input_prompt.format(user_id, order_history_str)
        target_text = target.format(target_item)

        return input_text, target_text


class SequentialSideInfoTask(Task):
    templates_dict = {
        0: PromptTarget(
            input_prompt="sequential recommendation - {}: \n\n"
                         "Predict for the user the next element of the following sequence -> {} \n"
                         "The category of each element of the sequence is -> {}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="sequential recommendation - {}: \n\n"
                         "Predict the next element which the user will buy given the following order history -> {} \n"
                         "Each item bought belongs to these categories (in order) -> {}",
            target_text="{}"
        ),
        2: PromptTarget(
            input_prompt="sequential recommendation - {}: \n\n"
                         "What is the element that should be recommended to the user knowing that it has "
                         "bought -> {} \n"
                         "Categories of the items are -> {}",
            target_text="{}"
        ),
        3: PromptTarget(
            input_prompt="sequential recommendation - {}: \n\n"
                         "Recommend to the user an item from the catalog given its order history -> {} \n"
                         "Each item of the order history belongs to the following categories (in order) -> {}",
            target_text="{}"
        ),
        4: PromptTarget(
            input_prompt="sequential recommendation - {}: \n\n"
                         "This is the order history of the user -> {} \n"
                         "These are the categories of each item -> {} \n"
                         "Please recommend the next element that the user will buy",
            target_text="{}"
        ),
        5: PromptTarget(
            input_prompt="sequential recommendation - {}: \n\n"
                         "Please predict what item is best to recommend to the user given its order history -> {} \n"
                         "Categories of each item -> {}",
            target_text="{}"
        )
    }

    def valid_templates(self, return_id: bool = True):
        return self.all_templates(return_id)

    def support_templates(self, return_id: bool = True):
        return []

    @Task.validate_args("user_id", "input_item_seq", "input_categories_seq", "target_item")
    def __call__(self, **kwargs):
        user_id = kwargs["user_id"]
        order_history = kwargs["input_item_seq"]
        input_categories_seq = kwargs["input_categories_seq"]
        target_item = kwargs["target_item"]

        # using all categories is maybe too much, let's use only one category for each item in the seq
        reduced_categories = [random.choice(categories) for categories in input_categories_seq]

        # random.choice applied to dict with int key returns a value
        input_prompt, target = random.choice(self.all_templates())

        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        order_history_str = separator.join(order_history)
        input_categories_str = separator.join(reduced_categories)

        input_text = input_prompt.format(user_id, order_history_str, input_categories_str)
        target_text = target.format(target_item)

        return input_text, target_text


class DirectTask(Task):
    templates_dict = {
        0: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Pick an item from the catalog likely to be bought by the user",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Recommend an item to the user",
            target_text="{}"
        ),
        2: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "What is the item that should be recommended to the user?",
            target_text="{}"
        ),
        3: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Select an item to present to the user",
            target_text="{}"
        ),
        4: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Please recommend an item that the user will buy",
            target_text="{}"
        ),
        5: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Please predict what item is best to recommend to the user",
            target_text="{}"
        )
    }

    def valid_templates(self, return_id: bool = True):
        return self.all_templates(return_id)[:5]

    def support_templates(self, return_id: bool = True):
        return []

    @Task.validate_args("user_id", "input_item_seq", "target_item")
    def __call__(self, **kwargs):
        user_id = kwargs["user_id"]
        input_item_seq = kwargs["input_item_seq"]
        target_item = kwargs["target_item"]

        if self.training:
            input_item_seq = input_item_seq + [target_item]

            target_idx = random.randint(0, len(input_item_seq) - 1)

            target_item = input_item_seq.pop(target_idx)

        # random.choice applied to dict with int key returns a value
        input_prompt, target = random.choice(self.all_templates())

        input_text = input_prompt.format(user_id)
        target_text = target.format(target_item)

        return input_text, target_text



class DirectSideInfoTask(Task):
    templates_dict = {
        0: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Pick an item from the catalog knowing that these are the categories "
                         "the user likes -> {}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Recommend an item to the user. The categories of the items bought by the user are -> {}",
            target_text="{}"
        ),
        2: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "What is the item that should be recommended to the user? It likes "
                         "these categories -> {}",
            target_text="{}"
        ),
        3: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Select an item to present to the user given the categories that it likes -> {}",
            target_text="{}"
        ),
        4: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "These are the categories of the items bought by the user -> {} \n"
                         "Please recommend an item that the user will buy",
            target_text="{}"
        ),
        5: PromptTarget(
            input_prompt="direct recommendation - {}: \n\n"
                         "Please predict what item is best to recommend to the user. The categories that it likes "
                         "are -> {}",
            target_text="{}"
        )
    }

    def valid_templates(self, return_id: bool = True):
        return self.all_templates(return_id)[:5]

    def support_templates(self, return_id: bool = True):
        return []

    @Task.validate_args("user_id", "input_item_seq", "input_categories_seq", "target_item", "target_categories")
    def __call__(self, **kwargs):
        user_id = kwargs["user_id"]
        input_item_seq = kwargs["input_item_seq"]
        input_categories_seq = kwargs["input_categories_seq"]
        target_categories = kwargs["target_categories"]
        target_item = kwargs["target_item"]

        if self.training:
            input_item_seq = input_item_seq + [target_item]
            input_categories_seq = input_categories_seq + [target_categories]

            target_idx = random.randint(0, len(input_item_seq) - 1)

            target_item = input_item_seq.pop(target_idx)
            target_categories = input_categories_seq.pop(target_idx)

        # we use only unique categories
        unique_categories = np.unique(list(itertools.chain.from_iterable(input_categories_seq)))

        # random.choice applied to dict with int key returns a value
        input_prompt, target = random.choice(self.all_templates())

        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        categories_liked_str = separator.join(unique_categories)

        input_text = input_prompt.format(user_id, categories_liked_str)
        target_text = target.format(target_item)

        return input_text, target_text