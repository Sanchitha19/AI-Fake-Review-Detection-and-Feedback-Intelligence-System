import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ReviewClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def extract_topics(self, reviews):
        """
        Cluster reviews and extract top words for each cluster.
        """
        if len(reviews) < self.n_clusters:
            # Handle case with fewer reviews than clusters
            return ["No significant clusters"]

        X = self.vectorizer.fit_transform(reviews)
        self.kmeans.fit(X)
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get cluster centers and top words
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        
        topics = []
        for i in range(self.n_clusters):
            top_words = [feature_names[ind] for ind in order_centroids[i, :5]]
            topics.append(f"Cluster {i+1}: " + ", ".join(top_words))
            
        return topics

if __name__ == "__main__":
    sample_reviews = [
        "The battery life is amazing, lasts all day.",
        "Battery drains too fast, need to charge every 2 hours.",
        "Awesome screen quality and colors.",
        "The display is pixelated and dull.",
        "The keyboard feel is great, very tactile.",
        "Keyboard keys are sticking, bad build quality."
    ]
    
    clustering = ReviewClustering(n_clusters=3)
    topics = clustering.extract_topics(sample_reviews)
    
    print("Detected Topics/Clusters:")
    for topic in topics:
        print(topic)
