import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List


def preprocess_data(texts: List[str], remove_stopwords=True, lemmatize=True) -> List[str]:
    """
    Preprocess a list of texts.
    """
    processed_texts = []
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Stopwords
    stop_words = set(stopwords.words('english'))

    for text in texts:
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

        # Remove punctuation and numbers
        text = re.sub(r'\p{P}+|[\d]+', '', text)

        # Tokenization
        tokens = text.split()

        # Stopword removal
        if remove_stopwords:
            tokens = [token for token in tokens if token not in stop_words]

        # Lemmatization
        if lemmatize:
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        processed_texts.append(" ".join(tokens))
    
    return processed_texts
