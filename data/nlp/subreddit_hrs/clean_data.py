import pandas as pd

CHUNK_SIZE = 512

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd


def split_data(row):
    eid, values = row
    print(values.genre)
    # Tokenize the input text and genre
    input_ids = tokenizer(values.text).input_ids
    genre_tokens = tokenizer(values.genre).input_ids
    
    # Remove the last token of genre_tokens (EOS token) as it will be doubled
    genre_tokens = genre_tokens[:-1]
    n = len(genre_tokens)  # Length of the genre tokens
    
    # Adjust CHUNK_SIZE to fit the genre at the start and end
    adjusted_chunk_size = CHUNK_SIZE - 2 * n
    
    chunked = []
    for chunk_start in range(0, len(input_ids), adjusted_chunk_size):
        # Extract the current chunk of the main text, leaving space for genre at start/end
        chunk = input_ids[chunk_start: chunk_start + adjusted_chunk_size]
        
        # Prepend and append the genre tokens to the chunk
        chunk_with_genre = genre_tokens + chunk + genre_tokens
        
        # If the final chunk is shorter than the adjusted_chunk_size, we still fill it to CHUNK_SIZE
        if len(chunk_with_genre) < adjusted_chunk_size:
            # # This could involve padding, depending on how you want to handle shorter final chunks
            padding = [tokenizer.pad_token_id] * (CHUNK_SIZE - len(chunk_with_genre))
            chunk_with_genre = chunk_with_genre + padding
            # continue
        
        chunked.append(chunk_with_genre)
    
    decoded_chunked = [tokenizer.decode(c, skip_special_tokens=True) for c in chunked]
    df = pd.DataFrame({'id': [eid] * len(chunked),
                       'genre': [values.genre]* len(chunked),
                       'decoded_text': decoded_chunked})
    return df
                         
def build_chunk_dataframe(text_data, cores=10):
    with Pool(cores) as p:
        chunks = list(tqdm(p.imap_unordered(split_data, text_data.iterrows()),
                            total=len(text_data)))
    
    return pd.concat(chunks)

def clean_non_unique(data):
    nunique_ids = (data.id.value_counts() > 1)
    nunique_ids = nunique_ids[nunique_ids].index
    return data[data.id.isin(nunique_ids)]


print('Load data reddit_crossgenre_as_csv.csv')
blog_corpus = pd.read_csv("data/nlp/subreddit_hrs/reddit_crossgenre_as_csv.csv")

blog_corpus.text = blog_corpus.text.apply(lambda x: str(x).strip())
full_blog_corpus = blog_corpus.groupby(['genre', 'id'])['text'].agg(lambda x: '<\s>'.join(x)).reset_index()

chunked_blog_data = build_chunk_dataframe(full_blog_corpus)
nunique_blog_data = clean_non_unique(chunked_blog_data)
print(nunique_blog_data.head(10))

nunique_blog_data.to_csv("data/nlp/subreddit_hrs/reddit_crossgenre_as_csv_processed.csv", index=False)



# # =========
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

# # ==========================
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