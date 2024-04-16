import torch
import numpy as np
# import faiss

import random
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval
import warnings

def join_text(list_text):
    return ' '.join(list_text)

class ContrastDataset(Dataset):
    def __init__(self, text_data, steps, batch_size=512, **kwargs):
        self.text_data = text_data
        
        self.batch_size =  batch_size#int(2 ** np.floor(np.log(batch_size)/np.log(2)))
        self.steps = steps * batch_size
        
        
        self.orig_authors = text_data.id.unique().tolist()
        self.n_authors = len(self.orig_authors)        
        self.text_data = self.text_data.set_index(['id', 'unique_id'])
        self.authors = self.orig_authors#self.populate_authors()
        
    # def populate_authors(self):
    #     n = self.steps//self.batch_size
    #     if self.steps%self.batch_size != 0:
    #         n += 1
    #     expanded = self.orig_authors * n

    #     return expanded[:self.steps]

    def __len__(self):
        return len(self.authors)

class BalancedContrastDataset(ContrastDataset):
    def generate_dataset_balance(self):
        keys = []
        for author in self.orig_authors:
            k, _ = author.split('_')
            if k not in keys:
                keys.append(k)
        
        count_dict = {k: 0 for k in keys}
        for author in self.orig_authors:
            k, _ = author.split('_')
            count_dict[k] += 1

        balanced_authors = []
        for author in self.orig_authors:
            k, _ = author.split('_')
            balanced_authors.append(1/(len(keys)*count_dict[k]))        

        return balanced_authors

    def populate_authors(self):
        balanced_orig_authors = self.generate_dataset_balance()
        n = self.steps//self.batch_size
        if self.steps%self.batch_size != 0:
            n += 1
        self.authors = []
        for _ in range(n):
            next_authors = np.random.choice(self.orig_authors, self.batch_size, p=balanced_orig_authors).tolist()
            self.authors += next_authors
            
        return self.authors[:self.steps]

class TextContrastDataset(ContrastDataset):
    # def __getitem__(self, i):
    #     auth = self.authors[i]
    #     # Check if the author has only one example
    #     if len(self.text_data.loc[auth]) == 1:
    #         warnings.filterwarnings("once", "Your dataset contains examples where an author only has 1 sample. Anchor and Replica will be set to the same document. ")
    #         warnings.warn("Your dataset contains examples where an author only has 1 sample. Anchor and Replica will be set to the same document. ")
    #         # Handling when only one text is available
    #         # You might choose to handle this differently based on your requirements
    #         anchor = replica = self.text_data.loc[auth].decoded_text.iloc[0]
    #     else:
    #         # Safe to sample two different texts as the author has more than one example
    #         anchor, replica = self.text_data.loc[auth].sample(2).decoded_text.tolist()

    #     return anchor, replica
    
    def __getitem__(self, i):
        
        auth = self.authors[i]
        anchor, replica = self.text_data.loc[auth].sample(2).decoded_text.tolist()

        return anchor, replica

import random

class MultiSampleTextContrastDataset(ContrastDataset):
    """
    A dataset class for creating text contrast samples with multiple documents.
    
    This class allows for sampling multiple documents as anchors and replicas for contrastive learning,
    where each anchor and replica consist of concatenated texts from multiple documents.
    The goal is to enhance learning by comparing varying but related texts.
    
    Attributes:
        sample_range (tuple): The range of the number of documents to sample for both anchor and replica.
                              The actual number of samples for each is randomly chosen within this range,
                              but the total will not exceed the available documents for an author.
    """
        
    def __init__(self, *args, **kwargs):
        """
        Initializes the MultiSampleTextContrastDataset with the dataset parameters.
        
        Args:
            *args: Variable length argument list to pass to the superclass.
            **kwargs: Arbitrary keyword arguments. 'sample_range' can be specified here.
        """
        super().__init__(*args, **kwargs)
        self.sample_range = tuple(kwargs.get("sample_range", (1, 1)))  # Expecting a tuple for range
        
    def __getitem__(self, i):
        """
        Retrieves the anchor and replica samples for a given index, with independent randomization of sample sizes.
        The method ensures no overlap between anchor and replica samples.

        Args:
            i (int): The index of the author for which to generate the samples.

        Returns:
            tuple: A tuple containing the concatenated anchor and replica texts.
        """
        auth = self.authors[i]
        total_docs = len(self.text_data.loc[auth])
        # Ensure the sampling does not exceed the total available documents
        available_samples = total_docs // 2

        # Adjust the sample range based on the half of the total documents available
        max_samples = min(self.sample_range[1], available_samples)
        min_samples = min(self.sample_range[0], max_samples)  # Ensure min samples less than max

        # Randomly determine the number of samples for anchor and replica within adjusted range
        anchor_n_samples = random.randint(min_samples, max_samples)
        replica_n_samples = random.randint(min_samples, max_samples)

        # Sample documents for anchor and replica
        all_docs = self.text_data.loc[auth].sample(n=anchor_n_samples + replica_n_samples)
        anchor_docs = all_docs.iloc[:anchor_n_samples]
        replica_docs = all_docs.iloc[anchor_n_samples:anchor_n_samples + replica_n_samples]

        # Concatenate texts for anchor and replica
        anchor_text = " ".join(anchor_docs.decoded_text.tolist())
        replica_text = " ".join(replica_docs.decoded_text.tolist())

        return anchor_text, replica_text




class TextSupervisedContrastDataset(ContrastDataset):
    def __init__(self, text_data, steps, batch_size=512, views=16):
        super().__init__(text_data, steps, batch_size)
        self.author_to_ordinal = {k: v for v, k in enumerate(self.orig_authors)}
        self.views = views

    def __getitem__(self, i):
        auth = self.authors[i]
        anchor = self.text_data.loc[auth].sample(self.views, replace=True).decoded_text.tolist()
        if len(anchor) == 1:
            anchor = anchor[0]

        return anchor, self.author_to_ordinal[auth]

# class TextHardNegativeContrastDataset(ContrastDataset):
#     def __init__(self, text_data, steps, batch_size=512, num_negatives=16, npz_path="data/nlp/blog_corpus/blog_train.tinybert_embeddings.npz"):
#         super().__init__(text_data, steps, batch_size)
#         self.author_to_ordinal = {k: v for v, k in enumerate(self.orig_authors)}
#         self.num_negatives = num_negatives

#         print("Building FAISS index for training data embeddings...")
#         self.embeddings = np.load(npz_path)['tinybert_embeddings']
#         d = self.embeddings.shape[-1]
#         self.index = faiss.IndexFlatL2(d)
#         self.index.add(self.embeddings)
#         print("Done building FAISS index!")

#     def __getitem__(self, i):

#         auth = self.authors[i]

#         anchor = self.text_data.loc[auth].sample(2)
#         anchor_text, replica_text = anchor.decoded_text.tolist()

#         # filter out any document ids of anchor author from negatives candidate pool
#         filter_out_idxs = set(map(lambda x: int(x.split("_")[0]), self.text_data.loc[auth].index.get_level_values(0).tolist()))
#         filter_idxs = np.array(list(set(range(self.index.ntotal)).difference(filter_out_idxs)), dtype=np.int64)

#         # NOTE: requires python3.9, faiss1.7.3, cuda11.7, all installed via conda
#         idx_selector = faiss.IDSelectorArray(filter_idxs)
        
#         # search FAISS index for top-num_negatives texts to anchor text
#         anchor_unique_id = int(anchor.index[0].split("_")[0])  # e.g. document "55061_blogs", use 55061 to index embeddings for FAISS
#         query = np.expand_dims(self.embeddings[anchor_unique_id], 0)
#         _, negatives_idxs = self.index.search(query, self.num_negatives, params=faiss.SearchParametersIVF(sel=idx_selector))
        
#         # get texts of negative examples based on retrieved indices
#         negatives_idxs = negatives_idxs[0]  # assume anchor index will be most similar result, first in negatives_idxs
#         negatives_idxs = list(map(lambda x: (str(x) + "_blogs").strip(), negatives_idxs))
#         negatives = self.text_data.swaplevel(0, 1).loc[negatives_idxs]  # search among "unique_id" field, which signify document
#         negatives_texts = negatives.decoded_text.tolist()
#         negatives_authors = negatives.index.get_level_values(1).tolist()  # get "id" fields of negatives, which signify author
        
#         try:
#             assert auth not in negatives_authors  # TODO this might not always be the case!
#         except AssertionError:
#             print(auth)
#             print(negatives_authors)

#         texts = [anchor_text, replica_text] + negatives_texts  # return as a single list

#         return texts, self.author_to_ordinal[auth]


class TextGenreSamplingContrastDataset(ContrastDataset):
    def __init__(self, text_data, steps, batch_size=512):
        super().__init__(text_data, steps, batch_size)
        
    # def populate_authors(self):
    #     n = self.steps//self.batch_size
    #     if self.steps%self.batch_size != 0:
    #         n += 1
    #     self.authors = []
    #     for _ in range(n):
    #         next_authors = np.random.choice(self.orig_authors, self.batch_size).tolist()
    #         self.authors += next_authors
            
    #     return self.authors[:self.steps]

    def __getitem__(self, i):
        auth = self.authors[i]

        # Selecting an anchor document
        author_data = self.text_data.loc[auth]
        anchor_data = author_data.sample(1)
        anchor = anchor_data.decoded_text.iloc[0]
        anchor_genre = anchor_data.genre.iloc[0]

        # Ensures that atleast 1/num_genre times we sample from the same genre to eliminate the possibility of learning that anchor.genre cannot equal replica.genre
        if random.randrange(0,len(self.text_data.genre.unique().tolist())):
            # Filtering for different genre within the author's data
            different_genre_df = author_data[author_data.genre != anchor_genre]
            if different_genre_df.empty:
                # warnings.warn(f"No different genre available for author {auth}. Sampling from the same genre.")
                replica = author_data.sample(1).decoded_text.iloc[0]
            else:
                # Sampling a replica from the filtered DataFrame
                replica = different_genre_df.sample(1).decoded_text.iloc[0]
        else:
            same_genre_df = author_data[(author_data.genre == anchor_genre) & (author_data.decoded_text != anchor)]
            if same_genre_df.empty:
                # warnings.warn(f"No different same available for author {auth}. Sampling from all examples.")
                replica = author_data.sample(1).decoded_text.iloc[0]
            else:
                # Sampling a replica from the filtered DataFrame
                replica = same_genre_df.sample(1).decoded_text.iloc[0]

        # If no different genre available for this author, print a warning
        

        return anchor, replica

class TextCollator:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __call__(self, texts):
        anchors, replicas = list(zip(*texts))
        
        
        config = dict(padding='longest',
                      return_tensors='pt',
                      truncation=True,
                      max_length=self.max_len,)
        
        encoded_anchors = self.tokenizer(list(anchors), **config)
        encoded_replicas = self.tokenizer(list(replicas), **config)
        
        return (encoded_anchors.input_ids,
                encoded_anchors.attention_mask,
                encoded_replicas.input_ids,
                encoded_replicas.attention_mask,
                )
    
class SupervisedTextCollator:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, pretokenized_texts):
        anchors, labels = list(zip(*pretokenized_texts))
        
        config = dict(padding='longest',
                      return_tensors='pt',
                      truncation=True,
                      max_length=self.max_len,)
        id_list, mask_list = [], []
        for views in anchors:
            encoded = self.tokenizer(list(views), **config)
            id_list.append(encoded.input_ids)
            mask_list.append(encoded.attention_mask)

        return (torch.stack(id_list, dim=0),
                torch.stack(mask_list, dim=0),
                torch.Tensor(labels)
                )

def build_dataset(dataframe,
                  steps=2000,
                  tokenizer=None,
                  max_len=128,
                  batch_size=16, 
                  num_workers=4, 
                  prefetch_factor=4,
                  shuffle=False,
                  num_to_multisample=[1,1] 
                  ):

    collator = TextCollator(tokenizer, max_len=max_len)
    dataset = MultiSampleTextContrastDataset(dataframe, steps, batch_size=batch_size, sample_range=num_to_multisample)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor,
                      collate_fn=collator,
                      drop_last=True,)
@RuntimeWarning
def build_supervised_dataset(dataframe,
                            steps,
                            tokenizer=None,
                            max_len=128,
                            batch_size=16,
                            views=16, 
                            num_workers=4, 
                            prefetch_factor=4,
                            shuffle=True):
    collator = SupervisedTextCollator(tokenizer, max_len=max_len)
    dataset = TextSupervisedContrastDataset(dataframe, steps, batch_size=batch_size, views=views)
    
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor,
                      collate_fn=collator,
                      drop_last=True,)

def build_genre_sampling_dataset(dataframe,
                                 steps,
                                 tokenizer=None,
                                 max_len=128,
                                 batch_size=16,
                                 num_workers=4,
                                 prefetch_factor=4,
                                 shuffle=True,):
    collator = TextCollator(tokenizer, max_len=max_len)
    dataset = TextGenreSamplingContrastDataset(dataframe, steps, batch_size=batch_size)
    
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor,
                    #   num_workers=0,
                    #   prefetch_factor=None,
                      collate_fn=collator,
                      drop_last=True,)



