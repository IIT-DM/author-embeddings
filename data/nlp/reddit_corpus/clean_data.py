import pandas as pd

CHUNK_SIZE = 512

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd





def split_data(row):
    eid, values = row
    input_ids = tokenizer(values.text).input_ids
    chunked = [input_ids[chunk: chunk + CHUNK_SIZE] for chunk in range(0, len(input_ids), CHUNK_SIZE)]
    decoded_chunked = tokenizer.batch_decode(chunked)
    df = pd.DataFrame({'id': [eid]*len(chunked),
                         'pretokenized_text': chunked,
                         'decoded_text': decoded_chunked})
    return df
                         
def build_chunk_dataframe(text_data, metadata=None, cores=10):
    with Pool(cores) as p:
        chunks = list(tqdm(p.imap_unordered(split_data, text_data.iterrows()),
                            total=len(text_data)))
    
    if metadata is not None:
        return pd.concat(chunks).merge(metadata, on='id')
    else:
        return pd.concat(chunks)

def clean_non_unique(data):
    nunique_ids = (data.id.value_counts() > 1)
    nunique_ids = nunique_ids[nunique_ids].index
    return data[data.id.isin(nunique_ids)]


print('Load data blog_as_csv.csv')
blog_corpus = pd.read_csv("data/nlp/blog_corpus/blog_as_csv.csv")


blog_corpus.text = blog_corpus.text.apply(lambda x: x.strip())
clean_blog_corpus = blog_corpus[['id', 'text']].groupby("id").agg(lambda x: '<\s>'.join(x))
meta_blog_corpus = blog_corpus[['id', 'age', 'topic', 'gender']].groupby("id").agg(lambda x: list(x)[0])
full_blog_corpus = meta_blog_corpus.merge(clean_blog_corpus, on='id')
print(full_blog_corpus.head(10))


chunked_blog_data = build_chunk_dataframe(full_blog_corpus, meta_blog_corpus)
nunique_blog_data = clean_non_unique(chunked_blog_data)
print(nunique_blog_data.head(10))

nunique_blog_data.to_csv("data/nlp/blog_corpus/blog_as_csv_preprocessed_deberta.csv", index=False)



=========
# def filter_text_length(row):
#     # Approximating the token count by word count
#     return len(row['text'].split()) > 0

# print('Load data reddit_crossgenre_as_csv.csv')
# blog_corpus = pd.read_csv("data/nlp/subreddit_hrs/reddit_crossgenre_as_csv.csv")

# # Filter based on text length
# filtered_blog_corpus = blog_corpus[blog_corpus.apply(filter_text_length, axis=1)]

# # Select and rename columns
# final_blog_data = filtered_blog_corpus[['id', 'text', 'genre']].rename(columns={'text': 'decoded_text'})

# # Sort by 'id'
# final_blog_data_sorted = final_blog_data.sort_values(by='id')

# print(final_blog_data_sorted.head(10))

# final_blog_data_sorted.to_csv("/share/lvegna/Repos/author/authorship-embeddings/data/nlp/subreddit_hrs/reddit_crossgenre_as_csv_processed.csv", index=False)

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