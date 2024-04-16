import os
import sys
import csv
import json





csv.field_size_limit(sys.maxsize)


def load_iarpa_5_genre_data(use_queries_for_training="exclude_query_docs"):

   
    CANDIDATES = "iarpa5/hrs_release_12-01-23/hrs_release_crossGenre_milestone_1.2-dryrun/mode_crossGenre/TA1/hrs_milestone_1.2-dryrun-samples_crossGenre-combined/data/hrs_milestone_1.2-dryrun-samples_crossGenre-combined_TA1_input_candidates.jsonl"
    QUERIES = "iarpa5/hrs_release_12-01-23/hrs_release_crossGenre_milestone_1.2-dryrun/mode_crossGenre/TA1/hrs_milestone_1.2-dryrun-samples_crossGenre-combined/data/hrs_milestone_1.2-dryrun-samples_crossGenre-combined_TA1_input_queries.jsonl"
    GROUND_TRUTH = "iarpa5/hrs_release_12-01-23/hrs_release_crossGenre_milestone_1.2-dryrun/mode_crossGenre/TA1/hrs_milestone_1.2-dryrun-samples_crossGenre-combined/groundtruth/hrs_milestone_1.2-dryrun-samples_crossGenre-combined_TA1_groundtruth.jsonl"
    def read_jsonl_file(fpath):
        with open(fpath, "r") as f:
            data = [json.loads(l.strip()) for l in f.readlines()]
        return data
    
    candidate_data = read_jsonl_file(CANDIDATES)
    query_data = read_jsonl_file(QUERIES)
    ground_truth = read_jsonl_file(GROUND_TRUTH)
    # print(ground_truth[0])
    ground_truth = {x["documentID"]: x["authorIDs"] for x in ground_truth}
    for x in candidate_data:
        x["authorIDs"] = ground_truth.get(x["documentID"])
    for x in query_data:
        x["authorIDs"] = ground_truth.get(x["documentID"])

    return query_data, candidate_data




def load_data(dataset_id, k: int=5, train_ratio: float=0.8, iarpa_use_queries_for_training: str="exclude_query_docs"):

    query_data, candidate_data = load_iarpa_5_genre_data(use_queries_for_training=iarpa_use_queries_for_training)


    return None, None, query_data, candidate_data
