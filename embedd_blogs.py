import csv
import jsonlines
import embedder
import numpy as np
from tqdm import tqdm
# Initialize your embedder
E = embedder.Embedder(model_path="/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-03-21_17-05-33.ckpt")

# Batch processing size
batch_size = 64

# Function to process a batch of rows
def process_batch(rows):
    texts = [row["decoded_text"] for row in rows]
    embeddings = E(texts)  # Assuming embedder can take a list of texts
    for i, row in enumerate(rows):
        row["embedding"] = embeddings[i].tolist()  # Convert numpy array to list
    return rows

# File paths
input_csv = "data/nlp/blog_corpus/blog_test.csv"
output_jsonl = "data/nlp/blog_corpus/blog_test_with_embeddings_best.jsonl"

# Process and write to jsonlines in batches
with open(input_csv, mode='r', encoding='utf-8') as infile, jsonlines.open(output_jsonl, mode='w') as outfile:
    reader = csv.DictReader(infile)
    batch = []
    for row in tqdm(reader):
        batch.append(row)
        if len(batch) == batch_size:
            processed_batch = process_batch(batch)
            outfile.write_all(processed_batch)
            batch = []
    # Process the last batch if it's not empty
    if batch:
        processed_batch = process_batch(batch)
        outfile.write_all(processed_batch)

print("Processing complete. Data saved to", output_jsonl)
