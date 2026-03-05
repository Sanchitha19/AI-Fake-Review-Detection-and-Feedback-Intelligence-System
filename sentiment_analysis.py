from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(review_text):
    """
    Analyze the sentiment of a review text using VADER.
    Returns:
        A dictionary containing positive, neutral, negative, and compound scores.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review_text)
    
    return {
        "positive": scores['pos'],
        "neutral": scores['neu'],
        "negative": scores['neg'],
        "compound": scores['compound']
    }

if __name__ == "__main__":
    test_text = "This product is absolutely amazing! I love the quality and it works perfectly."
    sentiment_results = analyze_sentiment(test_text)
    
    print(f"Review: {test_text}")
    print("\nSentiment Scores:")
    for key, value in sentiment_results.items():
        print(f" - {key.capitalize()}: {value}")
