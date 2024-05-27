import random

import numpy as np
import torch

from src.data.abstract_task import Template, LaikaTask, TaskOutput
from src.evaluate.metrics.error_metrics import ErrorMetric
from src.evaluate.metrics.ranking_metrics import RankingMetric


class P5RatingTask(LaikaTask):
    templates_dict = {
        "1-1": Template(
            input_text_placeholder="Which star rating will user_{user_id} give item_{item_id} ? "
                                   "( 1 being lowest and 5 being highest )",
            target_text_placeholder="{star_rating}"
        ),
        "1-2": Template(
            input_text_placeholder="How will user_{user_id} rate this product : {item_title} ? "
                                   "( 1 being lowest and 5 being highest )",
            target_text_placeholder="{star_rating}"
        ),
        "1-3": Template(  # support
            input_text_placeholder="Will user_{user_id} give item_{item_id} a {star_rating}-star rating ? "
                                   "( 1 being lowest and 5 being highest )",
            target_text_placeholder="{yes_no}"
        ),
        "1-4": Template(  # support
            input_text_placeholder="Does user_{user_id} like or dislike item_{item_id} ?",
            target_text_placeholder="{like_dislike}"
        ),
        "1-5": Template(
            input_text_placeholder="Predict the user_{user_id} 's preference on item_{item_id} ( {item_title} ) \n "
                                   "-1 \n -2 \n -3 \n -4 \n -5",
            target_text_placeholder="{star_rating}"
        ),
        "1-6": Template(
            input_text_placeholder="What star rating do you think {user_name} will give item_{item_id} ? "
                                   "( 1 being lowest and 5 being highest )",
            target_text_placeholder="{star_rating}"
        ),
        "1-7": Template(
            input_text_placeholder="How will {user_name} rate this product : {item_title} ? "
                                   "( 1 being lowest and 5 being highest )",
            target_text_placeholder="{star_rating}"
        ),
        "1-8": Template(  # support
            input_text_placeholder="Will {user_name} give a {star_rating}-star rating for {item_title} ? "
                                   "( 1 being lowest and 5 being highest )",
            target_text_placeholder="{yes_no}"
        ),
        "1-9": Template(  # support
            input_text_placeholder="Does {user_name} like or dislike {item_title} ?",
            target_text_placeholder="{like_dislike}"
        )
    }

    @classmethod
    def is_ranking_task(cls) -> bool:
        return False

    @classmethod
    def compatible_metrics(cls):
        return [ErrorMetric]

    # just in case evaluation is carried out on training tasks
    def inference_templates(self, return_id: bool = False):
        inference_templates_ids = ["1-1", "1-2", "1-5", "1-6", "1-7"]
        return [self.all_templates(return_id)[i] for i in inference_templates_ids]

    def _gaussian_sampling(self, og_rating: str):
        if self.training:
            if int(og_rating) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0 + 1.4) / 2),
                                                    std=torch.tensor((1.4 - 1.0) / 4)).item(), 1)
            elif int(og_rating) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5 + 2.4) / 2),
                                                    std=torch.tensor((2.4 - 1.5) / 4)).item(), 1)
            elif int(og_rating) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5 + 3.4) / 2),
                                                    std=torch.tensor((3.4 - 2.5) / 4)).item(), 1)
            elif int(og_rating) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5 + 4.4) / 2),
                                                    std=torch.tensor((4.4 - 3.5) / 4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5 + 5.0) / 2),
                                                    std=torch.tensor((5.0 - 4.5) / 4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return og_rating

    @staticmethod
    def _sample_wrong_rating(actual_rating: str):
        candidates_wrong_ratings = ["1", "2", "3", "4", "5"]
        candidates_wrong_ratings.remove(actual_rating)
        wrong_rating = random.choice(candidates_wrong_ratings)

        return wrong_rating

    def __call__(self, user_id: str, user_name: str, user_asin: str,
                 gt_item: list[str], gt_rating: list[str], gt_title: list[str],
                 **kwargs):

        [target_item] = gt_item
        [target_rating] = gt_rating
        [target_title] = gt_title

        # in case the user_name is not known, p5 uses the user asin code
        if user_name == "":
            user_name = user_asin

        # in case the item_title is not known, p5 uses "unknown title" as title
        if target_title == "":
            target_title = "unknown title"

        # the rating is perturbed for augmentation
        perturbed_rating = self._gaussian_sampling(target_rating)

        sampled_key = random.choice(self.all_templates(return_id=True))

        input_text_placeholder, target_text_placeholder = self.templates_dict[sampled_key]

        input_text = None
        target_text = None
        match sampled_key:

            case "1-1":
                input_text = input_text_placeholder.format(user_id=user_id, item_id=target_item)
                target_text = target_text_placeholder.format(star_rating=perturbed_rating)

            case "1-2":
                input_text = input_text_placeholder.format(user_id=user_id, item_title=target_title)
                target_text = target_text_placeholder.format(star_rating=perturbed_rating)

            case "1-3":
                rand_prob = random.random()
                if rand_prob > 0.5:
                    input_text = input_text_placeholder.format(user_id=user_id,
                                                               star_rating=target_rating,
                                                               item_id=target_item)
                    target_text = target_text_placeholder.format(yes_no="yes")
                else:
                    input_text = input_text_placeholder.format(user_id=user_id,
                                                               star_rating=self._sample_wrong_rating(target_rating),
                                                               item_id=target_item)
                    target_text = target_text_placeholder.format(yes_no="no")

            case "1-4":
                like_dislike = "like" if int(target_rating) >= 4 else "dislike"

                input_text = input_text_placeholder.format(user_id=user_id, item_id=target_item)
                target_text = target_text_placeholder.format(like_dislike=like_dislike)

            case "1-5":
                input_text = input_text_placeholder.format(user_id=user_id, item_id=target_item, item_title=target_title)
                target_text = target_text_placeholder.format(star_rating=perturbed_rating)

            case "1-6":
                input_text = input_text_placeholder.format(user_name=user_name, item_id=target_item)
                target_text = target_text_placeholder.format(star_rating=perturbed_rating)

            case "1-7":
                input_text = input_text_placeholder.format(user_name=user_name, item_title=target_title)
                target_text = target_text_placeholder.format(star_rating=perturbed_rating)

            case "1-8":
                rand_prob = random.random()
                if rand_prob > 0.5:
                    input_text = input_text_placeholder.format(user_name=user_name,
                                                               star_rating=target_rating,
                                                               item_title=target_title)
                    target_text = target_text_placeholder.format(yes_no="yes")
                else:
                    input_text = input_text_placeholder.format(user_name=user_name,
                                                               star_rating=self._sample_wrong_rating(target_rating),
                                                               item_title=target_title)
                    target_text = target_text_placeholder.format(yes_no="no")

            case "1-9":
                like_dislike = "like" if int(target_rating) >= 4 else "dislike"

                input_text = input_text_placeholder.format(user_name=user_name, item_title=target_title)
                target_text = target_text_placeholder.format(like_dislike=like_dislike)

        return [TaskOutput(input_text, target_text, ground_truth_for_eval=[target_rating])]


class P5EvalRatingTask(LaikaTask):
    templates_dict = {
        # prompt seen during training
        "1-6": P5RatingTask.templates_dict["1-6"],

        # new unseen prompt
        "1-10": Template(
            input_text_placeholder="Predict {user_name} 's preference towards {item_title} "
                                   "( 1 being lowest and 5 being highest )",
            target_text_placeholder="{star_rating}"
        )
    }

    @classmethod
    def is_ranking_task(cls) -> bool:
        return False

    @classmethod
    def compatible_metrics(cls):
        return [ErrorMetric]

    def inference_templates(self, return_id: bool = False):
        return self.all_templates(return_id)

    def __call__(self, user_id: str, user_name: str, user_asin: str,
                 gt_item: list[str], gt_rating: list[str], gt_title: list[str],
                 **kwargs):

        [target_item] = gt_item
        [target_rating] = gt_rating
        [target_title] = gt_title

        # in case the user_name is not known, p5 uses the user asin code
        if user_name == "":
            user_name = user_asin

        # in case the item_title is not known, p5 uses "unknown title" as title
        if target_title == "":
            target_title = "unknown title"

        # this is just for generality: the evaluator re-instantiates this task by forcing only one template
        # at a time, so both templates will be evaluated singularly
        sampled_key = random.choice(self.all_templates(return_id=True))

        input_text_placeholder, target_text_placeholder = self.templates_dict[sampled_key]

        input_text = None
        target_text = None
        match sampled_key:

            case "1-6":
                input_text = input_text_placeholder.format(user_name=user_name, item_id=target_item)
                target_text = target_text_placeholder.format(star_rating=target_rating)

            case "1-10":
                input_text = input_text_placeholder.format(user_name=user_name, item_title=target_title)
                target_text = target_text_placeholder.format(star_rating=target_rating)

        return [TaskOutput(input_text, target_text, ground_truth_for_eval=[target_rating])]


class P5SequentialTask(LaikaTask):
    templates_dict = {

        "2-1": Template(
            input_text_placeholder="Given the following purchase history of user_{user_id} : \n "
                                   "{order_history} \n predict next possible item to be purchased by the user ?",
            target_text_placeholder="{target_item}"
        ),
        "2-2": Template(
            input_text_placeholder="I find the purchase history list of user_{user_id} : \n "
                                   "{order_history} \n I wonder what is the next item to recommend to the user . "
                                   "Can you help me decide ?",
            target_text_placeholder="{target_item}"
        ),
        "2-3": Template(
            input_text_placeholder="Here is the purchase history list of user_{user_id} : \n "
                                   "{order_history} \n try to recommend next item to the user",
            target_text_placeholder="{target_item}"
        ),
        "2-4": Template(
            input_text_placeholder="Given the following purchase history of {user_name} : \n "
                                   "{order_history} \n predict next possible item for the user",
            target_text_placeholder="{target_item}"
        ),
        "2-5": Template(
            input_text_placeholder="Based on the purchase history of {user_name} : \n "
                                   "{order_history} \n "
                                   "Can you decide the next item likely to be purchased by the user ?",
            target_text_placeholder="{target_item}"
        ),
        "2-6": Template(
            input_text_placeholder="Here is the purchase history of {user_name} : \n "
                                   "{order_history} \n What to recommend next for the user ?",
            target_text_placeholder="{target_item}"
        ),

        # Extractive QAs templates
        "2-7": Template(
            input_text_placeholder="Here is the purchase history of user_{user_id} : \n "
                                   "{order_history} \n "
                                   "Select the next possible item likely to be purchased by the user "
                                   "from the following candidates : \n {candidate_items}",
            target_text_placeholder="{target_item}"
        ),
        "2-8": Template(
            input_text_placeholder="Given the following purchase history of {user_name} : \n "
                                   "{order_history} \n What to recommend next for the user? "
                                   "Select one from the following items : \n {candidate_items}",
            target_text_placeholder="{target_item}"
        ),
        "2-9": Template(
            input_text_placeholder="Based on the purchase history of user_{user_id} : \n "
                                   "{order_history} \n "
                                   "Choose the next possible purchased item from the following candidates : \n "
                                   "{candidate_items}",
            target_text_placeholder="{target_item}"
        ),
        "2-10": Template(
            input_text_placeholder="I find the purchase history list of {user_name} : \n "
                                   "{order_history} \n I wonder which is the next item to recommend to the user . "
                                   "Try to select one from the following candidates : \n {candidate_items}",
            target_text_placeholder="{target_item}"
        ),

        # Pairwise prediction templates
        "2-11": Template(
            input_text_placeholder="user_{user_id} has the following purchase history : \n "
                                   "{order_history} \n does the user likely to buy {target_item} next ?",
            target_text_placeholder="{yes_no}"
        ),
        "2-12": Template(
            input_text_placeholder="According to {user_name} 's purchase history list : \n "
                                   "{order_history} \n Predict whether the user will purchase {target_item} next ?",
            target_text_placeholder="{yes_no}"
        ),
    }

    @classmethod
    def is_ranking_task(cls) -> bool:
        return True

    @classmethod
    def compatible_metrics(cls):
        return [RankingMetric]

    # just in case evaluation is carried out on training tasks
    def inference_templates(self, return_id: bool = False):
        return self.all_templates(return_id)[:6]

    def qa_templates(self, return_id: bool = False):
        qa_ids = ["2-7", "2-8", "2-9", "2-10"]
        return [self.templates_dict[i] for i in qa_ids] if not return_id else qa_ids

    def pairwise_templates(self, return_id: bool = False):
        pairwise_ids = ["2-11", "2-12"]
        return [self.templates_dict[i] for i in pairwise_ids] if not return_id else pairwise_ids

    def __call__(self, user_id: str, user_name: str, input_item_seq: list[str],
                 gt_item: list[str], gt_title: list[str], catalog_items: np.ndarray[str],
                 **kwargs):

        out_list = []

        [target_item] = gt_item

        sampled_key = random.choice(self.inference_templates(return_id=True))
        input_text_placeholder, target_text_placeholder = self.templates_dict[sampled_key]

        # random select of string separator for order history
        rand_prob = random.random()
        separator = " , " if rand_prob > 0.5 else " -> "
        order_history_str = separator.join(input_item_seq)

        input_text_inference = None
        match sampled_key:
            case "2-1" | "2-2" | "2-3":
                input_text_inference = input_text_placeholder.format(user_id=user_id,
                                                                     order_history=order_history_str)
            case "2-4" | "2-5" | "2-6":
                input_text_inference = input_text_placeholder.format(user_name=user_name,
                                                                     order_history=order_history_str)

        target_text_inference = target_text_placeholder.format(target_item=target_item)

        out_list.append(TaskOutput(input_text_inference, target_text_inference, ground_truth_for_eval=[target_item]))

        if self.training:
            input_text_qa, target_text_qa = self._create_input_target_qa(user_id, user_name, input_item_seq,
                                                                         catalog_items, separator, order_history_str,
                                                                         target_item)

            input_text_pair, target_text_pair = self._create_input_target_pairwise(user_id, user_name, catalog_items,
                                                                                   order_history_str, target_item)

            out_list.append(TaskOutput(input_text_qa, target_text_qa))
            out_list.append(TaskOutput(input_text_pair, target_text_pair))

        return out_list

    def _create_input_target_qa(self, user_id: str, user_name: str, input_item_seq: list[str],
                                catalog_items: np.ndarray[str], separator: str, order_history_str: str,
                                target_item: str):

        sampled_key = random.choice(self.qa_templates(return_id=True))
        input_text_placeholder_qa, target_text_placeholder_qa = self.templates_dict[sampled_key]

        # choose as candidates items with which the user did not interact
        candidate_num = 99
        all_possible_candidates = np.setdiff1d(catalog_items, np.array(input_item_seq + [target_item]))
        candidates = np.random.choice(all_possible_candidates, size=candidate_num, replace=False)

        candidates = np.append(candidates, target_item)
        np.random.shuffle(candidates)

        candidates_str = separator.join(candidates)

        if sampled_key == "2-7" or sampled_key == "2-9":
            input_text_qa = input_text_placeholder_qa.format(user_id=user_id,
                                                             order_history=order_history_str,
                                                             candidate_items=candidates_str)
        # it's "2-8" or "2-10"
        else:
            input_text_qa = input_text_placeholder_qa.format(user_name=user_name,
                                                             order_history=order_history_str,
                                                             candidate_items=candidates_str)
        target_text_qa = target_text_placeholder_qa.format(target_item=target_item)

        return input_text_qa, target_text_qa

    def _create_input_target_pairwise(self, user_id: str, user_name: str, catalog_items: np.ndarray[str],
                                      order_history_str: str, target_item: str):

        sampled_key = random.choice(self.pairwise_templates(return_id=True))
        input_text_placeholder_pairwise, target_text_placeholder_pairwise = self.templates_dict[sampled_key]

        if random.random() > 0.5:
            next_item = target_item
            target_text = "yes"
        else:
            all_possible_candidates = catalog_items[catalog_items != target_item]
            [next_item] = np.random.choice(all_possible_candidates, size=1)
            target_text = "no"

        if sampled_key == "2-11":
            input_text_pairwise = input_text_placeholder_pairwise.format(user_id=user_id,
                                                                         order_history=order_history_str,
                                                                         target_item=next_item)
        # it's "2-12"
        else:
            input_text_pairwise = input_text_placeholder_pairwise.format(user_name=user_name,
                                                                         order_history=order_history_str,
                                                                         target_item=next_item)

        target_text_pairwise = target_text_placeholder_pairwise.format(yes_no=target_text)

        return input_text_pairwise, target_text_pairwise


class P5EvalSequentialTask(LaikaTask):
    templates_dict = {

        "2-3": P5SequentialTask.templates_dict["2-3"],

        "2-13": Template(
            input_text_placeholder="According to the purchase history of {user_name} : \n "
                                   "{order_history} \n Can you recommend the next possible item to the user ?",
            target_text_placeholder="{target_item}"
        ),
    }

    @classmethod
    def is_ranking_task(cls) -> bool:
        return True

    @classmethod
    def compatible_metrics(cls):
        return [RankingMetric]

    def inference_templates(self, return_id: bool = False):
        return self.all_templates(return_id)

    def __call__(self, user_id: str, user_name: str, input_item_seq: list[str],
                 gt_item: list[str], gt_title: list[str], catalog_items: np.ndarray[str],
                 **kwargs):

        [target_item] = gt_item

        sampled_key = random.choice(self.inference_templates(return_id=True))
        input_text_placeholder_inference, target_text_placeholder_inference = self.templates_dict[sampled_key]

        # random select of string separator for order history
        rand_prob = random.random()
        separator = " , " if rand_prob > 0.5 else " -> "
        order_history_str = separator.join(input_item_seq)

        if sampled_key == "2-3":
            input_text_inference = input_text_placeholder_inference.format(user_id=user_id,
                                                                           order_history=order_history_str)
        else:
            input_text_inference = input_text_placeholder_inference.format(user_name=user_name,
                                                                           order_history=order_history_str)
        target_text_inference = target_text_placeholder_inference.format(target_item=target_item)

        return [TaskOutput(input_text_inference, target_text_inference, ground_truth_for_eval=[target_item])]


class P5DirectTask(LaikaTask):
    templates_dict = {
        "5-1": Template(
            input_text_placeholder="Will user_{user_id} likely to interact with item_{item_id} ?",
            target_text_placeholder="{yes_no}"
        ),
        "5-2": Template(
            input_text_placeholder="Shall we recommend item_{item_id} to {user_name} ?",
            target_text_placeholder="{yes_no}"
        ),
        "5-3": Template(
            input_text_placeholder="For {user_name}, do you think it is good to recommend {item_title} ?",
            target_text_placeholder="{yes_no}",
        ),
        "5-4": Template(
            input_text_placeholder="I would like to recommend some items for user_{user_id} . "
                                   "Is the following item a good choice ? \n {item_title}",
            target_text_placeholder="{yes_no}"
        ),
        "5-5": Template(
            input_text_placeholder="Which item of the following to recommend for {user_name} ? \n {candidate_items}",
            target_text_placeholder="{target_item}"
        ),
        "5-6": Template(
            input_text_placeholder="Choose the best item from the candidates to recommend for {user_name} ? \n "
                                   "{candidate_items}",
            target_text_placeholder="{target_item}"
        ),
        "5-7": Template(
            input_text_placeholder="Pick the most suitable item from the following list and recommend to "
                                   "user_{user_id} : \n {candidate_items}",
            target_text_placeholder="{target_item}"
        ),
    }

    @classmethod
    def is_ranking_task(cls) -> bool:
        return True

    @classmethod
    def compatible_metrics(cls):
        return [RankingMetric]

    # for this task, neither template is properly an "inference template" since all require
    # the information from the ground truth, but in P5 paper they use the templates with the "target_item" as
    # target text nevertheless during the evaluation phase, so we comply with this
    def inference_templates(self, return_id: bool = False):
        return self.all_templates(return_id)[4:]

    def support_templates(self, return_id: bool = False):
        return self.all_templates(return_id)[:4]

    def __call__(self, user_id: str, user_name: str, input_item_seq: list[str], items_meta_dict: dict,
                 gt_item: list[str], gt_title: list[str], catalog_items: np.ndarray[str],
                 **kwargs):

        out_list = []
        [target_item] = gt_item
        [target_title] = gt_title

        sampled_key = random.choice(self.inference_templates(return_id=True))
        input_text_placeholder_inference, target_text_placeholder_inference = self.templates_dict[sampled_key]

        # choose as candidates items with which the user did not interact
        bullet_list_wrong_size = 99
        all_possible_candidates = np.setdiff1d(catalog_items, np.array(input_item_seq + gt_item))
        candidates = np.random.choice(all_possible_candidates, size=bullet_list_wrong_size, replace=False)

        candidates = np.append(candidates, target_item)
        np.random.shuffle(candidates)

        candidates_str = " , ".join(candidates)

        if sampled_key == "5-5" or sampled_key == "5-6":
            input_text_inference = input_text_placeholder_inference.format(user_name=user_name,
                                                                           candidate_items=candidates_str)

        # it's "5-7"
        else:
            input_text_inference = input_text_placeholder_inference.format(user_id=user_id,
                                                                           candidate_items=candidates_str)

        target_text_inference = target_text_placeholder_inference.format(target_item=target_item)

        out_list.append(TaskOutput(input_text_inference, target_text_inference, ground_truth_for_eval=[target_item]))

        if self.training:

            input_text_support, target_text_support = self._create_input_target_support(user_id, user_name,
                                                                                        catalog_items, target_item,
                                                                                        target_title, items_meta_dict)

            out_list.append(TaskOutput(input_text_support, target_text_support))

        return out_list

    def _create_input_target_support(self, user_id: str, user_name: str,
                                     catalog_items: np.ndarray[str], target_item: str,
                                     target_title: str, items_meta_dict: dict):

        sampled_key = random.choice(self.support_templates(return_id=True))
        input_text_placeholder_support, target_text_placeholder_support = self.templates_dict[sampled_key]

        rand_prob = random.random()
        if rand_prob > 0.5:
            item_to_recommend = target_item
            item_title = target_title
            target_text = "yes"
        else:
            all_possible_candidates = catalog_items[catalog_items != target_item]
            [item_to_recommend] = np.random.choice(all_possible_candidates, size=1)
            item_title = items_meta_dict[item_to_recommend].get("title", "unknown title")
            target_text = "no"

        target_text_support = target_text_placeholder_support.format(yes_no=target_text)

        input_text_support = None
        match sampled_key:

            case "5-1":
                input_text_support = input_text_placeholder_support.format(user_id=user_id,
                                                                           item_id=item_to_recommend)

            case "5-2":
                input_text_support = input_text_placeholder_support.format(item_id=item_to_recommend,
                                                                           user_name=user_name)

            case "5-3":
                input_text_support = input_text_placeholder_support.format(user_name=user_name,
                                                                           item_title=item_title)

            case "5-4":
                input_text_support = input_text_placeholder_support.format(user_id=user_id,
                                                                           item_title=item_title)

        return input_text_support, target_text_support


class P5EvalDirectTask(LaikaTask):
    templates_dict = {
        "5-5": P5DirectTask.templates_dict["5-5"],

        "5-8": Template(
            input_text_placeholder="We want to make recommendation for user_{user_id} .  "
                                   "Select the best item from these candidates : \n {candidate_items}",
            target_text_placeholder="{target_item}"
        )
    }

    @classmethod
    def is_ranking_task(cls) -> bool:
        return True

    @classmethod
    def compatible_metrics(cls):
        return [RankingMetric]

    def inference_templates(self, return_id: bool = False):
        return self.all_templates(return_id)

    def __call__(self, user_id: str, user_name: str, input_item_seq: list[str], items_meta_dict: dict,
                 gt_item: list[str], gt_title: list[str], catalog_items: np.ndarray[str],
                 **kwargs):

        [target_item] = gt_item

        sampled_key = random.choice(self.all_templates(return_id=True))
        input_text_placeholder, target_text_placeholder = self.templates_dict[sampled_key]

        # choose as candidates items with which the user did not interact
        bullet_list_wrong_size = 99
        all_possible_candidates = np.setdiff1d(catalog_items, np.array(input_item_seq + gt_item))
        candidates = np.random.choice(all_possible_candidates, size=bullet_list_wrong_size, replace=False)

        candidates = np.append(candidates, target_item)
        np.random.shuffle(candidates)

        candidates_str = " , ".join(candidates)

        if sampled_key == "5-5":
            input_text = input_text_placeholder.format(user_name=user_name, candidate_items=candidates_str)
        else:
            input_text = input_text_placeholder.format(user_id=user_id, candidate_items=candidates_str)

        target_text = target_text_placeholder.format(target_item=target_item)

        return [TaskOutput(input_text, target_text, ground_truth_for_eval=[target_item])]
