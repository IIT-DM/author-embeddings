#split datasets in local_data by id
import pandas as pd
import numpy as np

def train_test_split_by_author(df, test_size=0.05, random_state=42):
    unique_authors = df.id.unique()
    test_authors = len(unique_authors) * test_size
    in_test = np.random.choice(unique_authors, int(test_authors), replace=False)

    return df[~df.id.isin(in_test)], df[df.id.isin(in_test)]


blog_data = pd.read_csv("data/nlp/blog_corpus/blog_as_csv_preprocessed_deberta.csv")
blog_train, blog_test = train_test_split_by_author(blog_data)
blog_train.to_csv("data/nlp/blog_corpus/blog_train_deberta.csv", index=False)
blog_test.to_csv("data/nlp/blog_corpus/blog_test_deberta.csv", index=False)


# reddit_data = pd.read_csv("data/nlp/reddit_corpus/reddit_as_csv_preprocessed.csv")
# reddit_train, reddit_test = train_test_split_by_author(reddit_data)
# reddit_train.to_csv("data/nlp/reddit_corpus/reddit_train.csv", index=False)
# reddit_test.to_csv("data/nlp/reddit_corpus/reddit_test.csv", index=False)