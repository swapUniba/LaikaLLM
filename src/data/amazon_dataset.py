import gzip
import itertools
import json
import os
import pickle
import random
from collections import Counter
from functools import cached_property
from typing import Literal, Dict

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from src import DATA_DIR, ExperimentConfig, PROCESSED_DATA_DIR, RAW_DATA_DIR


def parse(path):
    with gzip.open(path, 'r') as g:
        for raw_meta_dict in g:
            yield eval(raw_meta_dict)


class AmazonDataset:

    def __init__(self,
                 dataset_name: Literal['beauty', 'toys', 'sport'],
                 add_prefix: bool = False,
                 integer_ids: bool = False):

        self.dataset_name = dataset_name
        self.add_prefix = add_prefix
        self.integer_ids = integer_ids

        user_items, item_count = self._read_sequential()

        # we compute the relative frequency of each item,
        # i.e. how many times an item is chosen by users divided total number of total items chosen by users
        total_occurrences = np.sum(list(item_count.values()))
        self.rel_freq_items = np.array([
            n_occurrences / total_occurrences
            for n_occurrences in item_count.values()
        ])

        # read mapping between user string id (ABXMSBDSI) and user int idxs (331)
        with open(os.path.join(DATA_DIR, dataset_name, 'datamaps.json'), "r") as f:
            datamaps = json.load(f)

        self.user_id2idx = {str(key): str(val) for key, val in datamaps['user2id'].items()}
        self.item_id2idx = {str(key): str(val) for key, val in datamaps['item2id'].items()}
        self.user_idx2id = {str(key): str(val) for key, val in datamaps['id2user'].items()}
        self.item_idx2id = {str(key): str(val) for key, val in datamaps['id2item'].items()}

        # here we save meta information (the "content") about items.
        # We only save info about items which appear in the user profiles
        relevant_items_id = set(self.item_id2idx.keys())
        meta_dict = {}
        print("Extracting meta info...")
        for meta_content in parse(os.path.join(DATA_DIR, dataset_name, 'meta.json.gz')):
            item_id = meta_content.pop("asin")
            if item_id in relevant_items_id:
                item_idx = self.item_id2idx[item_id]
                meta_dict[item_idx] = meta_content
        print("Done!")

        df_dict = {
            "user_id": [],
            "item_sequence": [],
            "categories_sequence": [],
            "price_sequence": [],
            "imurl_sequence": [],
            "brand_sequence": []
        }

        for user_idx, item_list_idxs in tqdm(user_items.items(), desc="Creating tabular data..."):

            if self.integer_ids is True:
                user_col_repeated = [user_idx for _ in range(len(item_list_idxs))]
                item_col_value = item_list_idxs
            else:
                user_col_repeated = [self.user_idx2id[user_idx] for _ in range(len(item_list_idxs))]
                item_col_value = [self.item_idx2id[item_idx] for item_idx in item_list_idxs]

            df_dict["user_id"].extend(user_col_repeated)
            df_dict["item_sequence"].extend(item_col_value)

            categories_list = []
            price_list = []
            imurl_list = []
            brand_list = []
            for item_idx in item_list_idxs:
                item_categories = meta_dict[item_idx].get("categories", "!No category!")[0]
                price = meta_dict[item_idx].get("price", "!No price!")
                imurl = meta_dict[item_idx].get("imUrl", "!No imurl!")
                brand = meta_dict[item_idx].get("brand", "!No brand!")

                categories_list.append(item_categories)
                price_list.append(str(price))
                imurl_list.append(str(imurl))
                brand_list.append(str(brand))

            df_dict["categories_sequence"].extend(categories_list)
            df_dict["price_sequence"].extend(price_list)
            df_dict["imurl_sequence"].extend(imurl_list)
            df_dict["brand_sequence"].extend(brand_list)

        data_df = pd.DataFrame.from_dict(df_dict)

        if self.add_prefix:
            data_df["user_id"] = "user_" + data_df["user_id"]
            data_df["item_sequence"] = "item_" + data_df["item_sequence"]

        self.original_df = data_df
        self.train_df, self.val_df, self.test_df = self._split_data(data_df)

    @cached_property
    def all_users(self):
        return pd.unique(self.original_df["user_id"])

    @cached_property
    def all_items(self):
        return pd.unique(self.original_df["item_sequence"].explode())

    @staticmethod
    def _split_data(exploded_data_df: pd.DataFrame):

        groupby_obj = exploded_data_df.groupby(by=["user_id"])

        # train set will be divided into input and target at each epoch: we will sample
        # each time a different input sequence and target item for each user so to reduce chances of
        # overfitting and performing a sort of augmentation in real time
        train_set = groupby_obj.nth[:-3].groupby(by=["user_id"]).agg(list).reset_index()

        # since validation set and test set do not need sampling (they must remain constant in order to validate
        # and evaluate the model fairly across epochs), we split directly here data in input and target.
        # It would be better to validate and test using entirely unknown users, but
        # in this phase we adhere to evaluation protocol of authors

        # if sequence is -> 1 2 3 4 5 6 7 8, VAL SET will have
        # input_sequence: 1 2 3 4 5 6
        # target_item: 7
        input_val_set = groupby_obj.nth[:-2].rename(columns={"item_sequence": "input_item_seq",
                                                             "categories_sequence": "input_categories_seq",
                                                             "price_sequence": "input_price_seq",
                                                             "imurl_sequence": "input_imurl_seq",
                                                             "brand_sequence": "input_brand_seq"})
        input_val_set = input_val_set.groupby(by=["user_id"]).agg(list).reset_index()
        target_val_set = groupby_obj.nth[-2].rename(columns={"item_sequence": "target_item",
                                                             "categories_sequence": "target_categories",
                                                             "price_sequence": "target_price",
                                                             "imurl_sequence": "target_imurl",
                                                             "brand_sequence": "target_brand"})

        val_set = input_val_set.merge(target_val_set, on="user_id")

        # if sequence is -> 1 2 3 4 5 6 7 8, TEST SET will have
        # input_sequence: 1 2 3 4 5 6 7
        # target_item: 8
        input_test_set = groupby_obj.nth[:-1].rename(columns={"item_sequence": "input_item_seq",
                                                              "categories_sequence": "input_categories_seq",
                                                              "price_sequence": "input_price_seq",
                                                              "imurl_sequence": "input_imurl_seq",
                                                              "brand_sequence": "input_brand_seq"})
        input_test_set = input_test_set.groupby(by=["user_id"]).agg(list).reset_index()
        target_test_set = groupby_obj.nth[-1].rename(columns={"item_sequence": "target_item",
                                                              "categories_sequence": "target_categories",
                                                              "price_sequence": "target_price",
                                                              "imurl_sequence": "target_imurl",
                                                              "brand_sequence": "target_brand"})

        test_set = input_test_set.merge(target_test_set, on="user_id")

        return train_set, val_set, test_set

    @staticmethod
    def sample_train_sequence(sample):

        out_dict = {}

        # a sequence has at least 1 data point, but it can have more depending on the length of the sequence
        # We must ensure that at least an element can be used as test set (that's why -1)
        # in the "sliding_size" is included the target item
        sliding_size = random.randint(1, len(sample["item_sequence"]) - 1)

        start_index = random.randint(0, len(sample["item_sequence"]) - sliding_size - 1)  # -1 since we start from 0
        end_index = start_index + sliding_size

        out_dict["user_id"] = sample["user_id"]
        out_dict["input_item_seq"] = sample["item_sequence"][start_index:end_index]
        out_dict["input_categories_seq"] = sample["categories_sequence"][start_index:end_index]
        out_dict["input_price_seq"] = sample["price_sequence"][start_index:end_index]
        out_dict["input_imurl_seq"] = sample["imurl_sequence"][start_index:end_index]
        out_dict["input_brand_seq"] = sample["brand_sequence"][start_index:end_index]
        out_dict["target_item"] = sample["item_sequence"][end_index]
        out_dict["target_categories"] = sample["categories_sequence"][end_index]
        out_dict["target_price"] = sample["price_sequence"][end_index]
        out_dict["target_imurl"] = sample["imurl_sequence"][end_index]
        out_dict["target_brand"] = sample["brand_sequence"][end_index]

        return out_dict

    def _read_sequential(self):

        user_items = dict()

        with open(os.path.join(RAW_DATA_DIR, self.dataset_name, "sequential_data.txt")) as f:
            for user_item_sequence in f:
                # user_item sequence is in the form {user_id}, {item_id}, {item_id}, ... {item_id}
                item_sequence = [str(item_idx) for item_idx in user_item_sequence.split()]
                user_idx = str(item_sequence.pop(0))
                user_items[user_idx] = item_sequence

        # count occurrences of each time (we must flatten) the item sequences first
        item_count = dict(Counter(itertools.chain.from_iterable(user_items.values())))

        return user_items, item_count

    def get_hf_datasets(self, merge_train_val: bool = False) -> Dict[str, datasets.Dataset]:

        val_hf_dataset = None
        # if merge_train_val is True:
        #     cleaned_dataset: pd.DataFrame = pd.read_pickle(self.cleaned_dataset_path)
        #     cleaned_grouped = self._group_dataset(cleaned_dataset)
        #
        #     rows_to_add_train = cleaned_grouped[cleaned_grouped["case_id"].isin(self.val_df["case_id"])]
        #     self.train_df = pd.concat([self.train_df, rows_to_add_train])
        # else:
        #     val_hf_dataset = Dataset.from_pandas(self.val_df, split=datasets.Split.VALIDATION, preserve_index=False)

        train_hf_ds = Dataset.from_pandas(self.train_df, split=datasets.Split.TRAIN, preserve_index=False)
        val_hf_ds = Dataset.from_pandas(self.val_df, split=datasets.Split.VALIDATION, preserve_index=False)
        test_hf_ds = Dataset.from_pandas(self.test_df, split=datasets.Split.TEST, preserve_index=False)

        # we create a dataset dict containing each split
        dataset_dict = {
            "train": train_hf_ds,
            "validation": val_hf_ds,
            "test": test_hf_ds
        }

        return dataset_dict

    def save(self, output_path: str = os.path.join(PROCESSED_DATA_DIR, "amzn_dat.pkl")):

        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = os.path.join(PROCESSED_DATA_DIR, "amzn_dat.pkl")):

        with open(path, "rb") as f:
            obj = pickle.load(f)

        return obj


def data_main():

    add_prefix = ExperimentConfig.add_prefix_item_users
    integer_ids = ExperimentConfig.integer_ids

    ds = AmazonDataset(dataset_name="toys",
                       add_prefix=add_prefix,
                       integer_ids=integer_ids)

    ds.save()


if __name__ == "__main__":
    data_main()
