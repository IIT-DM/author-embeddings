import pandas as pd

CHUNK_SIZE = 512

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=True)

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
blog_corpus = pd.read_csv("data/nlp/blog_corpus/blog_as_csv_deberta.csv")


blog_corpus.text = blog_corpus.text.apply(lambda x: x.strip())
clean_blog_corpus = blog_corpus[['id', 'text']].groupby("id").agg(lambda x: '<\s>'.join(x))
meta_blog_corpus = blog_corpus[['id', 'age', 'topic', 'gender']].groupby("id").agg(lambda x: list(x)[0])
full_blog_corpus = meta_blog_corpus.merge(clean_blog_corpus, on='id')
print(full_blog_corpus.head(10))


chunked_blog_data = build_chunk_dataframe(full_blog_corpus, meta_blog_corpus)
nunique_blog_data = clean_non_unique(chunked_blog_data)
print(nunique_blog_data.head(10))

nunique_blog_data.to_csv("data/nlp/blog_corpus/blog_as_csv_preprocessed_deberta.csv", index=False)


