import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from preprocess import ReviewPreprocessor

class FeedbackAnalyzer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.preprocessor = ReviewPreprocessor()

    def analyze_feedback(self, reviews):
        """
        Takes a list of genuine reviews, clusters them, and identifies key issues/features.
        """
        if not reviews or len(reviews) < self.n_clusters:
            return {"error": "Insufficient reviews for clustering", "clusters": []}

        # 1. Preprocess the reviews
        cleaned_reviews = [self.preprocessor.clean_text(r) for r in reviews]
        
        # 2. Vectorize using TF-IDF
        X = self.vectorizer.fit_transform(cleaned_reviews)
        
        # 3. Apply KMeans clustering
        self.kmeans.fit(X)
        
        # 4. Identify top keywords per cluster
        feature_names = self.vectorizer.get_feature_names_out()
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        
        cluster_insights = []
        for i in range(self.n_clusters):
            top_words = [feature_names[ind] for ind in order_centroids[i, :7]]
            
            # Map reviews to clusters
            cluster_indices = np.where(self.kmeans.labels_ == i)[0]
            sampled_reviews = [reviews[idx] for idx in cluster_indices[:2]] # Provide examples
            
            cluster_insights.append({
                "cluster_id": i + 1,
                "key_terms": top_words,
                "representative_examples": sampled_reviews,
                "size": len(cluster_indices)
            })
            
        return {
            "total_reviews": len(reviews),
            "clusters": cluster_insights
        }

if __name__ == "__main__":
    # Sample genuine reviews representing product issues and feature requests
    genuine_feedback = [
        "The battery life is way too short, it barely lasts 4 hours.",
        "Really wish there was a dark mode option for the app.",
        "The screen flickers when the brightness is low.",
        "Charging takes forever, please support fast charging.",
        "The UI is a bit cluttered, hard to find the settings menu.",
        "Would love to see integration with Google Calendar.",
        "The device gets very hot during video calls.",
        "Awesome performance but the software needs a dark theme.",
        "Battery drains even when the device is idle.",
        "Please add a way to export data to CSV."
    ]
    
    analyzer = FeedbackAnalyzer(n_clusters=3)
    results = analyzer.analyze_feedback(genuine_feedback)
    
    print("--- Product Feedback Analysis ---")
    for cluster in results['clusters']:
        print(f"\nInsight Group {cluster['cluster_id']}:")
        print(f" - Key Terms: {', '.join(cluster['key_terms'])}")
        print(f" - Cluster Size: {cluster['size']} reviews")
        print(f" - Example: \"{cluster['representative_examples'][0]}\"")
