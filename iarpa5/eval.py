
from tqdm import trange
from tqdm import tqdm


from typing import List, Dict, Union, Tuple


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from iarpa5.load_data import load_data
from iarpa5.dataset import CollateFnAuthorshipTest, ContrastiveLearningAuthorshipDatasetTest
from iarpa5.standalone_metrics import Experiment



def decode(
    test_dataloader: DataLoader,
    model,
    is_part_model: bool = False
) -> Tuple[List, List]:
    
    model.eval()
    # print(1)
    # pbar = trange(len(test_dataloader))
    # pbar.set_description(f"Decoding")
    # print(2)
    EMBEDDINGS, AUTHORS, DOCIDS = [], [], []
    for i, (docs, authors, docIDs) in enumerate(tqdm(test_dataloader)):
        # try:
        with torch.cuda.amp.autocast():
            if not is_part_model:
                embeddings = model(docs).detach().cpu().numpy()
            else:
                embeddings = model(docs["input_ids"], docs["attention_mask"]).detach().cpu().numpy()
        # except RuntimeError as e:
        #     pass

        EMBEDDINGS.extend(embeddings)
        AUTHORS.extend(authors)
        DOCIDS.extend(docIDs)


    return EMBEDDINGS, AUTHORS, DOCIDS


def decode_qc_to_hf_dataset_dict(query_dataloader, candidate_dataloader, model, is_part_model=False):

    q_vectors, q_authors, q_docids = decode(query_dataloader, model, is_part_model=is_part_model)
    c_vectors, c_authors, c_docids = decode(candidate_dataloader, model, is_part_model=is_part_model)

    q_dataset = HFDataset.from_dict({
        "documentID": q_docids,
        "features": q_vectors,
    })
    c_dataset = HFDataset.from_dict({
        "documentID": c_docids,
        "features": c_vectors,
    })
    dataset_dict = HFDatasetDict()
    dataset_dict["queries"] = q_dataset
    dataset_dict["candidates"] = c_dataset

    # dataset_dict.save_to_disk(PATH_TO_SAVE)

    return dataset_dict


def eval_iarpa(query_dataloader, candidate_dataloader, model, is_part_model=False, log_to_wandb=False, distance="euclidean"):

    dataset_dict = decode_qc_to_hf_dataset_dict(query_dataloader, candidate_dataloader, model, is_part_model=is_part_model)
    E = Experiment(features_path=None,
                   ground_truth_path="iarpa5/hrs_release_12-01-23/hrs_release_crossGenre_milestone_1.2-dryrun/mode_crossGenre/TA1/hrs_milestone_1.2-dryrun-samples_crossGenre-combined/groundtruth/hrs_milestone_1.2-dryrun-samples_crossGenre-combined_TA1_groundtruth.jsonl", 
                   dataset=dataset_dict, 
                   metric=distance)
    instance_metrics, summary_metrics, plot_data = E.compute_metrics(author_ids=None)

    return summary_metrics


# Declare module-level variables
QUERY_DATA = None
CANDIDATE_DATA = None
QUERY_DATALOADER = None
CANDIDATE_DATALOADER = None

def initialize_data_loaders(tokenizer, device):
    print('Initializing data loaders')
    global QUERY_DATA, CANDIDATE_DATA, QUERY_DATALOADER, CANDIDATE_DATALOADER

    _, _, QUERY_DATA, CANDIDATE_DATA = load_data("iarpa")
    query_dataset = ContrastiveLearningAuthorshipDatasetTest(QUERY_DATA)
    candidate_dataset = ContrastiveLearningAuthorshipDatasetTest(CANDIDATE_DATA)
    QUERY_DATALOADER = DataLoader(query_dataset, shuffle=False, batch_size=8, collate_fn=CollateFnAuthorshipTest(tokenizer, device))
    CANDIDATE_DATALOADER = DataLoader(candidate_dataset, shuffle=False, batch_size=8, collate_fn=CollateFnAuthorshipTest(tokenizer, device))

def run_iarpa5_eval(model, device, distance="euclidean"):
    print('Running IARPA5 eval')
    global QUERY_DATALOADER, CANDIDATE_DATALOADER

    tokenizer = model.tokenizer
    # Check if the data loaders are initialized. If not, initialize them.
    if QUERY_DATALOADER is None or CANDIDATE_DATALOADER is None:
        initialize_data_loaders(tokenizer, device)

    return eval_iarpa(QUERY_DATALOADER, CANDIDATE_DATALOADER, model, is_part_model=True, log_to_wandb=True, distance=distance)
