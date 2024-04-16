import pandas as pd
import json
import py3langid as langid
from tqdm import tqdm
from itertools import chain
import concurrent.futures
import time
# Paths
# train_path = '/share/brozonoyer/AUTHOR/data/PUBLIC_DATASETS/reddit/data/iur_dataset/train.jsonl'
# valid_path = '/share/brozonoyer/AUTHOR/data/PUBLIC_DATASETS/reddit/data/iur_dataset/validation.jsonl'
# output_path = 'data/nlp/reddit_corpus/reddit_as_csv.csv'

# def process_line(line):
#     time_start = time.time()
#     ex = json.loads(line)
#     author = ex['author_id']
#     texts = ex['syms']
    
#     english_texts = []
#     if len(texts) < 100:
#         return None
#     for text in texts:
#         if len(text.split()) > 300:
#             try:
#                 lang, _ = langid.classify(text)
#                 if lang == 'en':
#                     english_texts.append({
#                         'unique_id': f'reddit_{author}',
#                         'text': text
#                     })
#             except Exception:
#                 continue
#     print(f"Processed {author} in {time.time() - time_start:.2f} seconds")
#     if len(english_texts) < 100:
#         return None
#     return english_texts

# data = []  # List to collect rows for DataFrame

# with open(train_path, 'r') as ft, open(valid_path, 'r') as fv:
#     with concurrent.futures.ProcessPoolExecutor(max_workers=31) as executor:
#         # Submit all jobs and get a list of futures
#         futures = [executor.submit(process_line, line) for line in chain(ft, fv)]
#         # Iterate over futures as they complete
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#             try:
#                 result = future.result()
#                 if result:
#                     data.extend(result)
#             except Exception as e:
#                 print(f"Failed due to {e}")
# # Convert list of dictionaries to DataFrame
# reddit_df = pd.DataFrame(data)

# # Save to CSV
# reddit_df.to_csv(output_path, index=False)
# print(reddit_df.head(5))




# File path to the JSON lines file
input_file_path = 'data/nlp/subreddit_hrs/corrected_2_reduced_compound_subreddit_hrs.jsonl'

# Output file path for the CSV
output_file_path = 'data/nlp/subreddit_hrs/reddit_crossgenre_as_csv.csv'

# List to store the modified data
modified_data = []


# def get_subreddit(text):
#     if "subreddit-changemyview" in text:
#         return "subreddit-changemyview"
#     elif "subreddit-legaladvice" in text:
#         return "subreddit-legaladvice"
#     elif "subreddit-AskHistorians" in text:
#         return "subreddit-AskHistorians"
#     elif "subreddit-Jokes" in text:
#         return "subreddit-Jokes"
#     print(text, " \n did not contain a valid subreddit")
    
# Read the JSON lines file
with open(input_file_path, 'r') as file:
    for i, line in enumerate(file):
        
        # Parse each line as a JSON object
        data = json.loads(line, strict=False)
        
        # Rename the keys and modify the author_id as required
        modified_record = {
            'genre': data['source'],
            'unique_id': f"reddit_crossgenre_{data['authorIDs'][0]}",
            'id':data['authorIDs'][0],
            'text': str(data['fullText'])
        }
        
        # Add the modified record to the list
        modified_data.append(modified_record)


# Convert the list to a DataFrame and save it as a CSV

df = pd.DataFrame(modified_data)
df.to_csv(output_file_path, index=False)

