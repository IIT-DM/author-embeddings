#split datasets in local_data by id
import pandas as pd
import numpy as np

def train_test_split_by_author(df, test_size=0.05, random_state=42):
    unique_authors = df.id.unique()
    test_authors = len(unique_authors) * test_size
    in_test = np.random.choice(unique_authors, int(test_authors), replace=False)
    # import pdb; pdb.set_trace()
    return df[~df.id.isin(in_test)], df[df.id.isin(in_test)]


hrs_data = pd.read_csv("data/nlp/hrs_corpus/hrs_as_csv.csv")
hrs_train, hrs_test = train_test_split_by_author(hrs_data)
hrs_train.to_csv("data/nlp/hrs_corpus/hrs_train.csv", index=False)
hrs_test.to_csv("data/nlp/hrs_corpus/hrs_test.csv", index=False)


# reddit_data = pd.read_csv("data/nlp/reddit_corpus/reddit_as_csv_preprocessed.csv")
# reddit_train, reddit_test = train_test_split_by_author(reddit_data)
# reddit_train.to_csv("data/nlp/reddit_corpus/reddit_train.csv", index=False)
# reddit_test.to_csv("data/nlp/reddit_corpus/reddit_test.csv", index=False)