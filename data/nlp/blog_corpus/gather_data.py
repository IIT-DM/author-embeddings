import pandas as pd
import json


# File path to the JSON lines file
input_file_path = 'data/nlp/blog_corpus/blogtext.csv'

# Output file path for the CSV
output_file_path = 'data/nlp/blog_corpus/blog_as_csv_deberta.csv'

import pandas as pd
import os

#read blogtext.csv from the data/nlp/blog_corpus directory
blog_data = pd.read_csv(input_file_path)

#replace unique id values with "blog_n" where n is a number beggining at 0
n_values = len(blog_data.id.unique())
author_mapping = {k: v for k, v in zip(blog_data.id.unique(), range(n_values))}

blog_data['id'] = blog_data['id'].apply(lambda x: 'blog_' + str(author_mapping[x]))

blog_data.to_csv(output_file_path, index=False)
