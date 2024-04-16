import pandas as pd

# Load the dataset
file_path = '/share/lvegna/Repos/author/authorship-embeddings/data/nlp/hrs_corpus/hrs_train.csv'
df = pd.read_csv(file_path)

# Get statistics on the counts of each id
id_counts = df['id'].value_counts()
stats = id_counts.describe()
print(stats)