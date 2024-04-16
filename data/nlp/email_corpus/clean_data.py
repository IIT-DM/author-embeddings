import pandas as pd

CHUNK_SIZE = 512

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd

def filter_text_length(row):
    # Approximating the token count by word count
    return len(row['text'].split()) > 0

print('Load data reddit_crossgenre_as_csv.csv')
blog_corpus = pd.read_csv("data/nlp/subreddit_hrs/reddit_crossgenre_as_csv.csv")

# Filter based on text length
filtered_blog_corpus = blog_corpus[blog_corpus.apply(filter_text_length, axis=1)]

# Select and rename columns
final_blog_data = filtered_blog_corpus[['id', 'text', 'genre']].rename(columns={'text': 'decoded_text'})

# Sort by 'id'
final_blog_data_sorted = final_blog_data.sort_values(by='id')

print(final_blog_data_sorted.head(10))

final_blog_data_sorted.to_csv("/share/lvegna/Repos/author/authorship-embeddings/data/nlp/subreddit_hrs/reddit_crossgenre_as_csv_processed.csv", index=False)

# ==========================
# print('Load data reddit_as_csv.csv')
# reddit_corpus = pd.read_csv("data/nlp/reddit_corpus/reddit_as_csv.csv")
# reddit_corpus = reddit_corpus.rename(columns={'unique_id': 'id'})
# print(reddit_corpus.head(10))

# non_string_entries = reddit_corpus[~reddit_corpus.text.apply(lambda x: isinstance(x, str))]

# # Filter out entries that are not of type string
# reddit_corpus = reddit_corpus[reddit_corpus.text.apply(lambda x: isinstance(x, str))]

# reddit_corpus.text = reddit_corpus.text.apply(lambda x: str(x).strip())
# clean_reddit_corpus = reddit_corpus[['id', 'text']].groupby("id").agg(lambda x: '<\s>'.join(x))
# # meta_blog_corpus = reddit_corpus[['id', 'age', 'topic', 'gender']].groupby("id").agg(lambda x: list(x)[0])
# # full_blog_corpus = meta_blog_corpus.merge(clean_blog_corpus, on='id')
# print(clean_reddit_corpus.head(10))


# chunked_reddit_data = build_chunk_dataframe(clean_reddit_corpus)
# nunique_reddit_data = clean_non_unique(chunked_reddit_data)
# print(nunique_reddit_data.head(10))

# nunique_reddit_data.to_csv("data/nlp/reddit_corpus/reddit_as_csv_preprocessed.csv", index=False)