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
    nunique_ids = (data.id.value_counts() > 5)
    nunique_ids = nunique_ids[nunique_ids].index
    return data[data.id.isin(nunique_ids)]

print('Load data hrs_release_08-14-23_crossGenre-combined_TA1_combined_preprocessed.csv')
hrs_corpus = pd.read_csv("data/nlp/hrs_corpus/hrs_preproccessed.csv")


hrs_corpus.text = hrs_corpus.text.apply(lambda x: x.strip())
clean_hrs_corpus = hrs_corpus[['id', 'text']].groupby("id").agg(lambda x: '<\s>'.join(x))
# meta_hrs_corpus = hrs_corpus[['id', 'age', 'topic', 'gender']].groupby("id").agg(lambda x: list(x)[0])
# full_hrs_corpus = meta_hrs_corpus.merge(clean_hrs_corpus, on='id')
print(clean_hrs_corpus.head(10))


# chunked_hrs_data = build_chunk_dataframe(full_hrs_corpus, meta_hrs_corpus)
chunked_hrs_data = build_chunk_dataframe(clean_hrs_corpus)
nunique_hrs_data = chunked_hrs_data#clean_non_unique(chunked_hrs_data)
print(nunique_hrs_data.head(10))

nunique_hrs_data.to_csv("data/nlp/hrs_corpus/hrs_as_csv.csv", index=False)


