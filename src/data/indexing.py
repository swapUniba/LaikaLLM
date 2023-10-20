from copy import deepcopy

import numpy as np
import numpy_indexed as npi
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util

from src import ExperimentConfig


def find_closest_clusters(km):

    cluster_idxs = np.arange(km.n_clusters)
    cluster_centers = deepcopy(km.cluster_centers_)

    # we start from 0 cluster index
    reordered_cluster_idxs = [0]

    # - 1 because the last cluster does not have other clusters to compare
    for i in range(len(cluster_idxs) - 1):

        cluster_idx = reordered_cluster_idxs[i]

        centroid = km.cluster_centers_[cluster_idx]
        cluster_centers[cluster_idx, :] = np.full_like(cluster_centers[cluster_idx], fill_value=np.nan)

        closest_cluster = np.nanargmax(util.cos_sim(centroid, cluster_centers), axis=1)
        reordered_cluster_idxs.append(closest_cluster.item())

    return reordered_cluster_idxs


def main_new_indexing(original_data_df: pd.DataFrame):

    seed = ExperimentConfig.random_seed

    unique_items_df = original_data_df.groupby(by="item_sequence").nth[0].reset_index(drop=True)

    # we want to use description as content indicator
    content_items = unique_items_df["description_sequence"]

    # for items for which we don't have description, we'll use title as content indicator
    missing_content_mask = content_items == ""
    content_items[missing_content_mask] = unique_items_df.loc[missing_content_mask, "title_sequence"]

    # for items for which we don't have description or title, we'll use categories as content indicator
    missing_content_mask = content_items == ""
    content_items[missing_content_mask] = unique_items_df.loc[missing_content_mask, "categories_sequence"].str.join(", ")

    df_to_process = pd.concat((unique_items_df["item_sequence"], content_items), axis=1).rename(columns={
        "item_sequence": "item",
        "description_sequence": "content"
    })

    encoder_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda:0")

    content_embs = encoder_model.encode(df_to_process["content"].tolist(), show_progress_bar=True)

    kmeans = KMeans(n_clusters=150, n_init="auto", random_state=seed)
    res = pd.Series(kmeans.fit_predict(content_embs), dtype=str)

    # re_ordered_clusters = find_closest_clusters(kmeans)

    old_idxs = df_to_process["item"]
    new_idxs = res.str.cat(old_idxs, sep="-")

    # start_new_idx = 1
    # old_idxs = df_to_process["item"].values
    # new_idxs = np.full_like(old_idxs, fill_value=np.nan)
    # for cluster_idx in re_ordered_clusters:
    #     mask_items = np.flatnonzero(res == cluster_idx)
    #     idx_to_assign = start_new_idx + np.arange(len(mask_items))
    #
    #     cat_cluster = np.char.add(np.array([f"{cluster_idx}/" for _ in range(len(idx_to_assign))]), idx_to_assign.astype(str))
    #
    #     new_idxs[mask_items] = cat_cluster
    #
    #     # + 1 so to avoid overlap between last item of a cluster and first item of the next cluster
    #     start_new_idx = idx_to_assign[-1] + 1

    # old_idxs = old_idxs.astype(str)
    # new_idxs = new_idxs.astype(str)

    # original df replacement
    original_df_mask = npi.indices(old_idxs, original_data_df["item_sequence"].values)
    new_column_original_df = new_idxs[original_df_mask].reset_index(drop=True)

    original_data_df["item_sequence"] = new_column_original_df

    return original_data_df
