import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class ReviewPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.lemmatizer = nltk.WordNetLemmatizer()

    def clean_text(self, text):
        """
        Clean text: lowercase, remove punctuation, remove stopwords, and lemmatize.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove punctuation and special characters using regex
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and word not in self.punctuation
        ]

        return " ".join(cleaned_tokens)

    def process_dataframe(self, df, text_column='review_text'):
        """
        Apply cleaning to a dataframe column.
        """
        df['cleaned_review'] = df[text_column].apply(self.clean_text)
        return df

if __name__ == "__main__":
    # Example usage/test
    sample_data = {
        'review_text': [
            "This product is amazing! I loved it so much.",
            "Worst purchase ever. Broke after 2 days!!!",
            "It's okay, not the best but works fine for the price.",
            "Fake review detection is important for e-commerce."
        ]
    }
    df = pd.DataFrame(sample_data)
    preprocessor = ReviewPreprocessor()
    df = preprocessor.process_dataframe(df)
    
    print("Original Reviews:")
    print(df['review_text'])
    print("\nCleaned Reviews:")
    print(df['cleaned_review'])
