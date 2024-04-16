import random
from collections import defaultdict
from itertools import chain

from torch.utils.data import Dataset


class CollateFnAuthorshipTrain(object):

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        docs1 = self.tokenizer([x[0] for x in batch], padding="longest", truncation=True, return_tensors="pt").to(self.device)
        docs2 = self.tokenizer([x[1] for x in batch], padding="longest", truncation=True, return_tensors="pt").to(self.device)
        return docs1, docs2


class SupervisedCollateFnAuthorshipTrain(object):

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        return self.tokenizer(list(chain(*batch)), padding="longest", truncation=True, return_tensors="pt").to(self.device)  # (bsz * n_views)


class CollateFnAuthorshipTest(object):

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        docs = self.tokenizer([x[0] for x in batch], padding="longest", truncation=True, return_tensors="pt").to(self.device)
        authors = [x[1] for x in batch]
        docIDs = [x[2] for x in batch]
        return docs, authors, docIDs


class ContrastiveLearningAuthorshipDatasetTrain(Dataset):

    def __init__(self, data, n_views=2):
        super().__init__()

        self.n_views = n_views

        author2docs = defaultdict(list)
        for x in data:
            author2docs[x["authorIDs"][0]].append(x["fullText"])

        # filter out authors with fewer than n_views docs from training set
        author2docs = dict(filter(lambda k_v: len(k_v[1]) >= n_views, author2docs.items()))

        self.authors = list(author2docs.keys())
        self.docs = []
        for author in self.authors:
            self.docs.append(author2docs[author])
        
    def __len__(self):
        return len(self.authors)

    def __getitem__(self, i):
        return random.sample(self.docs[i], self.n_views)  # sample 2/n_views docs for contrastive learning


class ContrastiveLearningAuthorshipDatasetTest(Dataset):

    def __init__(self, data):
        super().__init__()

        # Sorting the data based on the length of 'fullText'
        sorted_data = sorted(data, key=lambda x: -len(x["fullText"]))

        self.docs = []
        self.authors = []
        self.docIDs = []
        for x in sorted_data:
            self.docs.append(x["fullText"])
            self.authors.append(x["authorIDs"][0])
            self.docIDs.append(x["documentID"])
        
    def __len__(self):
        return len(self.authors)

    def __getitem__(self, i):
        return (self.docs[i], self.authors[i], self.docIDs[i])
