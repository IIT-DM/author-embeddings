import json
from tqdm import tqdm

# Path to the incorrectly formatted JSONL file
input_file_path = '/share/lvegna/Repos/author/authorship-embeddings/data/nlp/subreddit_hrs/2_reduced_compound_subreddit_hrs.jsonl'
# Specify the path for the corrected output file
output_file_path = '/share/lvegna/Repos/author/authorship-embeddings/data/nlp/subreddit_hrs/corrected_2_reduced_compound_subreddit_hrs.jsonl'

# Open the input file
with open(input_file_path, 'r') as input_file:
    # Read the entire content of the file, which is a single line
    single_line = input_file.read()
    lines = [r'{"documentID"'+e for e in single_line.split(r'{"documentID"') if e]
    
    with open(output_file_path, 'w') as output_file:
        for line in lines:
            line = json.loads(line)
            output_file.write(json.dumps(line)+'\n')
            
print(f"Corrected JSONL file has been saved to {output_file_path}")
