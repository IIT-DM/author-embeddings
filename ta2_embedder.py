# //////////////////////////////////////////////////////////////////
# // Charles River Analytics, Inc., Cambridge, Massachusetts
# // Copyright (C) 2023. All Rights Reserved.
# // Developed under the IARPA HIATUS program by the AUTHOR team.
# // Contact ccall@cra.com for questions regarding the code.
# //////////////////////////////////////////////////////////////////

import os
import torch
from model import *
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

class Embedder:
    """ class that takes in one HTS Document and returns its embedding vector as a ID numpy array"""

    def __init__(self):

        # model_path = "/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-02-29_17-20-36.ckpt"
        # model_path = "/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-03-06_16-16-48.ckpt" #Long_LSTM
        model_path = "/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-03-12_14-22-44.ckpt" #Attention sum pooler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ContrastiveLSTMAttentionHead.load_from_checkpoint(checkpoint_path=model_path,
                                                              map_location=torch.device(self.device))
        self.model.eval()
        self.tokenizer = self.model.tokenizer

    def get_embedding(self, hts_documents: list):
        """
        :param hts_documents:  List of HTSDocument objects
                            (will need languages list in the future)
        :return: 1D numpy array which is the embedding for all documents
        """
        chunk_len = self.model.chunk_length 
        concatenated_texts = ''
        
        for hts_document in hts_documents:
            # Convert full text to tokens
            tokens = self.tokenizer.encode(hts_document, add_special_tokens=False)
            
            # Sample a random start point for the 512 tokens
            max_start = max(0, len(tokens) - chunk_len)
            start = random.randint(0, max_start) if max_start > 0 else 0
            end = start + chunk_len
            
            # Slice the tokens and decode to text
            sampled_tokens = tokens[start:end]
            text = self.tokenizer.decode(sampled_tokens, skip_special_tokens=True)
            
            # Concatenate the decoded text
            concatenated_texts += ' ' + text  # Add space to separate documents
        
        # Tokenize the concatenated text
        tokenized = self.tokenizer(concatenated_texts, padding='do_not_pad', truncation=True, max_length=self.model.max_length, return_tensors="pt").to(self.device)
        
        # Pass the concatenated sequence to the model
        arr = self.model(tokenized['input_ids'], tokenized["attention_mask"]).detach().cpu().numpy().squeeze()
        return arr


# class Embedder:
#     """ class that takes in one HTS Document and returns its embedding vector as a ID numpy array"""

#     def __init__(self):

#         model_path = "/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-02-06_11-55-20.ckpt"
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = ContrastiveLSTMHead.load_from_checkpoint(checkpoint_path=model_path,
#                                                               map_location=torch.device(self.device))
#         self.model.eval()
#         self.tokenizer = self.model.tokenizer

#     def get_embedding(self, raw):
#         """
#         :param hts_document:  HTSDocument object
#                               (will need languages list in the future)
#         :return: 1D numpy array
#         """
#         self.tokenizer.model_max_length = 512
#         self.model.max_length = 512
#         tokenized = self.tokenizer(raw, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
#         arr = self.model(tokenized['input_ids'], tokenized["attention_mask"]).detach().cpu().numpy().squeeze()
#         return np.array(np.mean([arr], axis=1)).astype('float32')
    



def dist(x, y):
    return 1 - abs(np.linalg.norm(np.array(x) - np.array(y)) / 2)



if __name__ == "__main__":
    E = Embedder()
    pd.options.display.max_columns = None
    # df = pd.read_csv("/share/jdruce/dev/author_ml/author-part/data/nlp/hrs_corpus/hrs_train.csv")
    df = pd.read_csv("/share/lvegna/Repos/author/authorship-embeddings/data/nlp/blog_corpus/blog_test_deberta.csv")

    # Find unique IDs with more than 6 entries
    counts = df['id'].value_counts()
    valid_ids = counts[counts > 6].index[:50]  # Taking the first 50 unique IDs

    candidate_embeddings = []
    query_embeddings = []

    # Process texts for each valid ID
    for blog_id in tqdm(valid_ids):
        texts = df[df['id'] == blog_id]['decoded_text'].tolist()[:24]
        midpoint = len(texts) // 2
        candidate_texts = texts[:midpoint]
        query_texts = texts[midpoint:]

        candidate_embeddings.append(E.get_embedding(candidate_texts))
        query_embeddings.append(E.get_embedding(query_texts))

    # Calculate distances within and between authors
    intra_distances = []
    inter_distances = []

    for i in tqdm(range(len(candidate_embeddings))):
        intra_distances.append(dist(candidate_embeddings[i], query_embeddings[i]))

        for j in range(len(query_embeddings)):
            if i != j:
                inter_distances.append(dist(candidate_embeddings[i], query_embeddings[j]))

    # Calculate and print the average distances
    avg_intra_distance = np.mean(intra_distances)
    avg_inter_distance = np.mean(inter_distances)

    print("Average intra-author distance:", avg_intra_distance)
    print("Average inter-author distance:", avg_inter_distance)

