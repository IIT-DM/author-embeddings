import pandas as pd
import json

# File path to the JSON lines file
input_file_path = '/share/lvegna/Repos/author/authorship-embeddings/data/nlp/hrs_corpus/hrs_milestone_1.2-dryrun-samples_crossGenre-combined_TA1_input_combined.jsonl'
input_file_path_groundtruth = '/share/lvegna/Repos/author/authorship-embeddings/data/nlp/hrs_corpus/hrs_milestone_1.2-dryrun-samples_crossGenre-combined_TA1_groundtruth.jsonl'

# Output file path for the CSV
output_file_path = 'data/nlp/hrs_corpus/hrs_preproccessed.csv'

# List to store the modified data
modified_data = []

docid2author_dict = {}
unique_author_ids = set()  # Set to keep track of unique author IDs

with open(input_file_path_groundtruth, 'r') as file:
    for line in file:
        # Parse each line as a JSON object
        data = json.loads(line)
        # Assuming data['authorIDs'] is a list of author IDs for a single document
        docid2author_dict[str(data['documentID'])] = data['authorIDs']
        # Add author IDs to the set of unique author IDs
        unique_author_ids.update(data['authorIDs'])

# Print the number of unique author IDs
print(f"Number of unique author IDs: {len(unique_author_ids)}")

# Read the JSON lines file
with open(input_file_path, 'r') as file:
    for line in file:
        # Parse each line as a JSON object
        data = json.loads(line)
        # print(data)
        
        # Rename the keys and modify the author_id as required
        author_id = docid2author_dict[str(data['documentID'])][0]
        modified_record = {
            'unique_id': f"hrs_{author_id}",
            'id':author_id,
            'text': data['fullText']
        }
        
        # Add the modified record to the list
        modified_data.append(modified_record)


# Convert the list to a DataFrame and save it as a CSV

df = pd.DataFrame(modified_data)
df.to_csv(output_file_path, index=False)

