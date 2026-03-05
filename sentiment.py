from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_vader(self, text):
        """
        Get sentiment scores using VADER.
        """
        scores = self.vader_analyzer.polarity_scores(text)
        # compound score: -1 (very negative) to 1 (very positive)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        return label, compound

    def get_sentiment_textblob(self, text):
        """
        Get sentiment using TextBlob (Polarity and Subjectivity).
        """
        analysis = TextBlob(text)
        # Polarity: -1 to 1, Subjectivity: 0 to 1
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity

    def analyze_reviews(self, df, text_column='review_text'):
        """
        Batch analyze reviews in a dataframe.
        """
        df['sentiment_label'], df['sentiment_score'] = zip(*df[text_column].apply(self.get_sentiment_vader))
        df['polarity'], df['subjectivity'] = zip(*df[text_column].apply(self.get_sentiment_textblob))
        return df

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample_text = "The product is absolutely wonderful and life-changing!"
    label, score = analyzer.get_sentiment_vader(sample_text)
    print(f"VADER Sentiment: {label} (Score: {score})")
    
    pol, subj = analyzer.get_sentiment_textblob(sample_text)
    print(f"TextBlob: Polarity={pol:.2f}, Subjectivity={subj:.2f}")
