import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug('NLTK stopwords and punkt tokenizers downloaded successfully.')

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, 
    removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    try:
        logger.debug(f"Original text: {text}")

        text = text.lower()
        logger.debug(f"Lowercased text: {text}")

        text = word_tokenize(text)
        logger.debug(f"Tokenized text: {text}")

        text = [word for word in text if word.isalnum()]
        logger.debug(f"Alphanumeric filtered: {text}")

        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words and word not in string.punctuation]
        logger.debug(f"Stopwords and punctuation removed: {text}")

        text = [ps.stem(word) for word in text]
        logger.debug(f"Stemmed text: {text}")

        final_text = " ".join(text)
        logger.debug(f"Final transformed text: {final_text}")

        return final_text
    except Exception as e:
        logger.error("Error in transform_text: %s", e)
        raise

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates,
    and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
