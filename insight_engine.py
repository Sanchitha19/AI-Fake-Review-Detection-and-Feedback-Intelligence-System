import pandas as pd
from sentiment_analysis import analyze_sentiment
from preprocess import ReviewPreprocessor

class InsightEngine:
    def __init__(self):
        self.preprocessor = ReviewPreprocessor()
        
        # Simple keyword maps for classification
        self.complaint_keywords = [
            'broke', 'slow', 'expensive', 'bad', 'poor', 'terrible', 'worst', 
            'issue', 'problem', 'bug', 'crash', 'fail', 'error', 'difficult',
            'cluttered', 'heavy', 'waste', 'disappointed', 'flicker', 'drain'
        ]
        
        self.feature_request_keywords = [
            'wish', 'hope', 'would love', 'add', 'feature', 'support', 
            'integration', 'dark mode', 'theme', 'option', 'include', 'improve',
            'need', 'want', 'please', 'better', 'future', 'update'
        ]

    def _get_top_keywords(self, reviews, top_n=3):
        """Helper to find frequent keywords in a list of reviews."""
        all_words = []
        for r in reviews:
            cleaned = self.preprocessor.clean_text(r)
            all_words.extend(cleaned.split())
        
        if not all_words:
            return []
            
        freq = pd.Series(all_words).value_counts()
        return freq.head(top_n).index.tolist()

    def generate_insights(self, reviews):
        """
        Generate human-readable product insights from a list of reviews.
        """
        if not reviews:
            return {"error": "No reviews provided"}

        complaints = []
        feature_requests = []
        sentiments = []

        for r in reviews:
            r_lower = r.lower()
            sent = analyze_sentiment(r)
            sentiments.append(sent['compound'])
            
            # Simple heuristic for complaints (Negative sentiment + keywords)
            if sent['compound'] < 0 or any(kw in r_lower for kw in self.complaint_keywords):
                complaints.append(r)
                
            # Simple heuristic for features (Keywords like 'wish', 'add', 'request')
            if any(kw in r_lower for kw in self.feature_request_keywords):
                feature_requests.append(r)

        # Calculations
        neg_count = sum(1 for s in sentiments if s <= -0.05)
        neg_percent = (neg_count / len(reviews)) * 100 if reviews else 0
        
        top_complaint_themes = self._get_top_keywords(complaints)
        top_feature_themes = self._get_top_keywords(feature_requests)

        # Formatting human-readable insights
        insights = {
            "summary_metrics": {
                "total_reviews": len(reviews),
                "negative_review_percentage": f"{neg_percent:.1f}%",
                "average_sentiment": round(sum(sentiments)/len(sentiments), 4)
            },
            "top_complaints": [
                f"Issues related to: {', '.join(top_complaint_themes)}" if top_complaint_themes else "No significant complaints found."
            ],
            "most_requested_features": [
                f"Users are asking for improvements in: {', '.join(top_feature_themes)}" if top_feature_themes else "No clear feature requests identified."
            ]
        }
        
        return insights

if __name__ == "__main__":
    sample_reviews = [
        "The battery life is way too short, it drains in 2 hours.",
        "I wish there was a dark mode option for the app.",
        "The app keeps crashing when I open the settings.",
        "Please add support for Google Drive integration.",
        "The UI is terrible and very hard to navigate.",
        "Would be great to have a way to export data.",
        "Excellent product, but a bit expensive.",
        "I love the design but the screen flickers sometimes."
    ]
    
    engine = InsightEngine()
    results = engine.generate_insights(sample_reviews)
    
    print("--- Generated Insights ---")
    print(f"Negative Review %: {results['summary_metrics']['negative_review_percentage']}")
    print(f"\nComplaints: {results['top_complaints'][0]}")
    print(f"Feature Requests: {results['most_requested_features'][0]}")
