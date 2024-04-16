import pandas as pd
import py3langid as langid

def clean_and_save_csv(file_path):
    # Load the DataFrame from the specified file path
    df = pd.read_csv(file_path)
    print(len(df))
    
    # Function to check if text is English and longer than 300 characters
    def is_english_and_long(text):
        lang, _ = langid.classify(text)
        return lang == 'en' and len(text) >= 300

    # Apply the function to filter the DataFrame
    df = df[df['decoded_text'].apply(is_english_and_long)]

    # Construct the path for the cleaned file
    clean_file_path = file_path.replace('.csv', '_clean.csv')
    print(len(df))
    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(clean_file_path, index=False)
    print(f"Saved cleaned file to {clean_file_path}")

# File paths
train_dir = 'data/nlp/blog_corpus/blog_train.csv'
test_dir = 'data/nlp/blog_corpus/blog_test.csv'

# Clean and save the files
clean_and_save_csv(train_dir)
clean_and_save_csv(test_dir)
