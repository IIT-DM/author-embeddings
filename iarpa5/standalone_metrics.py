# minimal functionality to obtain TA1 metrics outside of a docker
# use default parameters

import argparse
import importlib
import logging
import os
import sys
from typing import Callable, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from scipy.stats import rankdata, hmean
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances
from yaml import Loader, load

from simplification.cutil import simplify_coords
from tqdm.auto import tqdm


def compute_rank_metrics(
    all_needle_ranks,
    n_candidates,
    top_ks=None,
    all_needle_distances=None,
    compute_metrics_at_distance=True,
    plot_resolution=100,
    disable_progress_bars=False,
    **kwargs,
):
    """Compute TA1 metrics."""
    instance = {}
    summary = {}
    plots = {}
    epsilon = 0.0001

    mpr = np.array([np.mean(hits) / n_candidates for hits in all_needle_ranks])
    instance["Mean Percentile Rank"] = mpr
    summary["Harmonic Mean of Mean Percentile Rank"] = hmean(mpr)

    min_ranks = np.array([min(hits) for hits in all_needle_ranks])
    instance["Rank of Nearest True Match"] = min_ranks
    summary["Mean Reciprocal Rank"] = 1 / hmean(min_ranks)

    all_n_needles = np.array([len(hits) for hits in all_needle_ranks])
    if top_ks is None:
        top_ks = [1, 8, "all"]

    if "all" in top_ks:
        delta = int(n_candidates / plot_resolution)
        if not delta:
            raise ValueError(
                f"Not enough candidates ({n_candidates}) to plot a resolution of {plot_resolution}. "
                "Please set 'plot_resolution' to a lower number."
            )
        ranks_ = list(range(1, n_candidates, delta))
        average_success_at_k, average_recall_at_k, average_precision_at_k, average_fpr_at_k = (
            [0.0],
            [0.0],
            [1.0],
            [0.0],
        )
        for k in tqdm(ranks_, desc="Computing metrics for many values of k", disable=disable_progress_bars):
            instance_, summary_ = rank_metrics_at_k(all_needle_ranks, all_n_needles, k, n_candidates)
            average_success_at_k.append(summary_[f"Average Success at {k}"])
            average_recall_at_k.append(summary_[f"Average Recall at {k}"])
            average_precision_at_k.append(summary_[f"Average Precision at {k}"])
            average_fpr_at_k.append(summary_[f"Average FPR at {k}"])
        average_success_at_k.append(1.0)
        average_recall_at_k.append(1.0)
        average_precision_at_k.append(all_n_needles.mean() / n_candidates)
        average_fpr_at_k.append(1.0)

        success_coords = [[x, y] for x, y in zip([0] + ranks_ + [n_candidates], average_success_at_k)]
        recall_coords = [[x, y] for x, y in zip([0] + ranks_ + [n_candidates], average_recall_at_k)]
        precision_coords = [[x, y] for x, y in zip([0] + ranks_ + [n_candidates], average_precision_at_k)]
        roc_coords = [[x, y] for x, y in zip(average_fpr_at_k, average_recall_at_k)]
        pvr_coords = [[x, y] for x, y in zip(average_recall_at_k, average_precision_at_k)]

        success_simplified = np.asarray(simplify_coords(success_coords, epsilon))
        recall_simplified = np.asarray(simplify_coords(recall_coords, epsilon))
        precision_simplified = np.asarray(simplify_coords(precision_coords, epsilon))
        roc_simplified = np.asarray(simplify_coords(roc_coords, epsilon))
        pvr_simplified = np.asarray(simplify_coords(pvr_coords, epsilon))

        plots["Average Success at k"] = {
            "x_values": list(success_simplified[:, 0]),
            "y_values": list(success_simplified[:, 1]),
        }
        plots["Average Recall at k"] = {
            "x_values": list(recall_simplified[:, 0]),
            "y_values": list(recall_simplified[:, 1]),
        }
        plots["Average Precision at k"] = {
            "x_values": list(precision_simplified[:, 0]),
            "y_values": list(precision_simplified[:, 1]),
        }
        plots["Average ROC"] = {"x_values": list(roc_simplified[:, 0]), "y_values": list(roc_simplified[:, 1])}
        plots["Average Precision vs. Recall"] = {
            "x_values": list(pvr_simplified[:, 0]),
            "y_values": list(pvr_simplified[:, 1]),
        }
        summary.update(
            {
                "Area Under ROC Curve": np.trapz(x=average_fpr_at_k, y=average_recall_at_k),
                "Harmonic Mean of Average Precision": hmean(average_precision_at_k),
            }
        )

    for k in top_ks:
        if k == "all":
            continue
        instance_, summary_ = rank_metrics_at_k(all_needle_ranks, all_n_needles, k, n_candidates)
        instance.update(instance_)
        summary.update(summary_)

    if compute_metrics_at_distance and all_needle_distances is not None:
        min_dist = np.array([min(hits) for hits in all_needle_distances])
        instance["Distance to Nearest True Match"] = min_dist
        summary["Mean Distance to Nearest True Match"] = min_dist.mean()

        max_dist = all_needle_distances.apply(max).max()
        delta = max_dist / plot_resolution
        dists_ = np.arange(delta, max_dist, delta)    # FIXME gives division by zero error
        average_success_at_k, average_recall_at_k = [0.0], [0.0]
        for dist in tqdm(dists_, desc="Computing metrics for many distances", disable=disable_progress_bars):
            # TODO refactor to pair rank and distance information to get the rest of the retrieval distance metrics
            # rank_metrics_at_k works for for success and recall out-of-the-box
            # but precision and FPR depend on the number of retrieved docs, which is not available here
            instance_, summary_ = rank_metrics_at_k(
                all_needle_distances, all_n_needles, dist, n_candidates, compute_at_distance=True
            )
            average_success_at_k.append(summary_[f"Average Success at {dist}"])
            average_recall_at_k.append(summary_[f"Average Recall at {dist}"])
        average_success_at_k.append(1.0)
        average_recall_at_k.append(1.0)

        dists_ = [0.0] + list(dists_) + [max_dist]

        success_coords = [[x, y] for x, y in zip(dists_, average_success_at_k)]
        recall_coords = [[x, y] for x, y in zip(dists_, average_recall_at_k)]

        success_simplified = np.asarray(simplify_coords(success_coords, epsilon))
        recall_simplified = np.asarray(simplify_coords(recall_coords, epsilon))

        plots["Average Success at Distance"] = {
            "x_values": list(success_simplified[:, 0]),
            "y_values": list(success_simplified[:, 1]),
        }
        plots["Average Recall at Distance"] = {
            "x_values": list(recall_simplified[:, 0]),
            "y_values": list(recall_simplified[:, 1]),
        }

    return pd.DataFrame(instance), pd.Series(summary, name="Summary"), plots


def rank_metrics_at_k(all_needle_ranks, all_n_needles, k, n_candidates, compute_at_distance=False):
    """Calculate TA1 metrics at a particular number of retrieved documents."""
    if not compute_at_distance and k > n_candidates:
        raise ValueError(f"Cannot compute metrics at k={k} with only {n_candidates} candidates.")
    instance = {}
    summary = {}
    instance["Total Possible Retrievals"] = all_n_needles

    hits_count = np.array([len([rank for rank in hits if rank <= k]) for hits in all_needle_ranks])
    instance[f"True Retreivals at {k}"] = hits_count

    success = (hits_count > 0).astype(float)
    instance[f"Success at {k}"] = success
    summary[f"Average Success at {k}"] = success.mean()

    recall = hits_count / all_n_needles
    instance[f"Recall at {k}"] = recall
    summary[f"Average Recall at {k}"] = recall.mean()

    precision = hits_count / k
    instance[f"Precision at {k}"] = precision
    summary[f"Average Precision at {k}"] = precision.mean()

    fpr = (k - hits_count) / (n_candidates - all_n_needles)
    instance[f"FPR at {k}"] = fpr
    summary[f"Average FPR at {k}"] = fpr.mean()

    return instance, summary


class Experiment():
    """Container for TA1 experiments.

    Attributes:
        config (Dict[str, Any]): Configuration for the experiment.
        dataset (datasets.DatasetDict):
        metric (Union[str, Callable]): The metric to use when calculating distance between instances in a
            feature array. If metric is a string, it must be one of the options
            allowed by scikit-learn's pairwise_distances function.
            Alternatively, if metric is a callable function, it is called on each
            pair of instances (rows) and the resulting value recorded. The callable
            should take two arrays from as input and return a value indicating
            the distance between them.
    """

    def __init__(self, features_path, ground_truth_path, dataset=None, metric="euclidean", *args, **kwargs):
        """Initialize a new experiment."""
        super().__init__(*args, **kwargs)

        self.metric = metric

        # Load features
        if dataset is not None:
            self.dataset = dataset
        else:  
            self.dataset = self.load_dataset(features_path)
        
        # Load labels
        mappings = pd.read_json(ground_truth_path, lines=True)
        assert "authorIDs" in mappings.columns, mappings.columns
        assert "documentID" in mappings.columns, mappings.columns
        mapping_dict = mappings.set_index("documentID")["authorIDs"].to_dict()

        q = pd.DataFrame(self.dataset["queries"])
        c = pd.DataFrame(self.dataset["candidates"])
        # Add authorIDs column
        q["authorIDs"] = q["documentID"].map(mapping_dict)
        c["authorIDs"] = c["documentID"].map(mapping_dict)
        q["authorIDs"] = q["authorIDs"].astype(str)
        c["authorIDs"] = c["authorIDs"].astype(str)

        self.candidate_labels = c["authorIDs"].tolist()
        self.query_labels = q["authorIDs"].tolist()

        self.dataset = DatasetDict({"queries": Dataset.from_pandas(q), "candidates": Dataset.from_pandas(c)})

        self.distance_matrix = None
        self.rank_and_distance = None

    def rank_relevant_documents(self):
        """Compute rank and distance information."""
        if not set(self.query_labels) <= set(self.candidate_labels):
            raise NotImplementedError("Authors in query set which aren't present in candidate set")

        print("Beginning pairwise distance calculation")
        self.distance_matrix = pairwise_distances(
            self.dataset["queries"]["features"],
            self.dataset["candidates"]["features"],
            metric=self.metric,
            n_jobs=1,
        )
        self.distance_matrix = self.distance_matrix.round(6)  # (#queries, #candidates)

        print("Ranking candidate documents based on distance from query documents.")
        ranks = rankdata(self.distance_matrix, method="ordinal", axis=1)  # (#queries, #candidates), the higher the entry means the greater the distance in the distance

        print("Finding rank and distance from queries to true candidates.")
        # Indices of candidate documents belonging to each author
        candidates_arr = np.array(self.candidate_labels)
        author_masks = {label: np.flatnonzero(candidates_arr == label) for label in set(self.candidate_labels)}  # {authorID: List[int], indices of docs in 0-(#candidates-1)}

        all_needle_ranks = []
        all_needle_distances = []
        for i, label in enumerate(self.query_labels):
            instance_needle_ranks = list(ranks[i, author_masks[label]])
            all_needle_ranks.append(instance_needle_ranks)
            # all_needle_ranks.append((np.arange(len(instance_needle_ranks)) + 1).tolist())

            instance_needle_distances = list(self.distance_matrix[i, author_masks[label]])
            all_needle_distances.append(instance_needle_distances)
            # all_needle_distances.append((np.zeros(len(instance_needle_distances), dtype=float) + 0.0001).tolist())

        self.rank_and_distance = pd.DataFrame(
            {
                "Ranks of Relevant Documents": all_needle_ranks,
                "Distance to Relevant Documents": all_needle_distances,
            },
            index=self.query_labels,
        )

    def compute_metrics(self, author_ids=None):
        """Compute TA1 metrics on a subset of query documents."""
        if self.rank_and_distance is None:
            self.rank_relevant_documents()

        if author_ids is None:
            rank_and_distance = self.rank_and_distance
        else:
            rank_and_distance = self.rank_and_distance.loc[author_ids, :]

        rank_and_distance = rank_and_distance.rename(
            columns={
                "Ranks of Relevant Documents": "all_needle_ranks",
                "Distance to Relevant Documents": "all_needle_distances",
            },
            errors="raise",
        )
        idx = rank_and_distance.index
        rank_and_distance = rank_and_distance.to_dict("series")

        instance_metrics, summary_metrics, plot_data = compute_rank_metrics(
            n_candidates=len(self.candidate_labels),
            top_ks=None,
            compute_metrics_at_distance=True,
            plot_resolution=100,
            disable_progress_bars=True,
            **rank_and_distance,
        )
        instance_metrics.index = idx
        return instance_metrics, summary_metrics, plot_data

    @staticmethod
    def load_dataset(dataset_path: Union[str, os.PathLike]) -> DatasetDict:
        """Load a datasets.DatasetDict from disk."""
        dataset = load_from_disk(str(dataset_path))
        if not isinstance(dataset, DatasetDict):
            raise ValueError("dataset_path should point to a DatasetDict, not a Dataset")
        if "queries" not in dataset.keys():
            raise ValueError("Dataset must contain 'queries' split")
        if "documentID" not in dataset["queries"].features.keys():
            raise ValueError("Queries dataset must contain 'documentID' column")
        if "features" not in dataset["queries"].features.keys():
            raise ValueError(f"Queries dataset must contain 'features' column: {dataset['queries'].features.keys()}")

        if "candidates" not in dataset.keys():
            raise ValueError("Dataset must contain 'candidates' split")
        if "documentID" not in dataset["candidates"].features.keys():
            raise ValueError("Candidates dataset must contain 'documentID' column")
        if "features" not in dataset["candidates"].features.keys():
            raise ValueError("Candidates dataset must contain 'features' column")
        return dataset


if __name__ == "__main__":
    ()