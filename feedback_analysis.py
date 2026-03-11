"""
feedback_analysis.py — KMeans topic clustering for genuine reviews.
Groups customer feedback into thematic clusters and surfaces the key terms
per cluster so product teams can see what topics dominate.
"""

from typing import Dict, List, Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import ReviewPreprocessor


class FeedbackAnalyzer:
    """
    Applies KMeans clustering to a list of genuine review texts,
    returning cluster summaries with key terms and example reviews.
    """

    def __init__(self, n_clusters: int = 4) -> None:
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=2000, stop_words="english", ngram_range=(1, 2)
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        self.preprocessor = ReviewPreprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_feedback(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Cluster `reviews` and return per-cluster key terms + examples.

        Returns dict with keys:
            total_reviews : int
            clusters      : list of cluster dicts
        """
        if not reviews:
            return {"error": "No reviews provided.", "clusters": []}

        # Dynamically reduce clusters if not enough samples
        effective_k = min(self.n_clusters, len(reviews))
        if effective_k < 2:
            return {
                "total_reviews": len(reviews),
                "clusters": [
                    {
                        "cluster_id": 1,
                        "key_terms": [],
                        "representative_examples": reviews[:2],
                        "size": len(reviews),
                    }
                ],
            }

        if effective_k != self.n_clusters:
            logger.warning(
                f"Only {len(reviews)} reviews — reducing clusters from "
                f"{self.n_clusters} to {effective_k}."
            )
            self.kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init="auto")

        # Preprocess
        cleaned = [self.preprocessor.clean_text(r) for r in reviews]

        # Vectorise
        X = self.vectorizer.fit_transform(cleaned)

        # Cluster
        self.kmeans.fit(X)

        feature_names = self.vectorizer.get_feature_names_out()
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]

        clusters: List[Dict[str, Any]] = []
        for i in range(effective_k):
            top_terms = [feature_names[idx] for idx in order_centroids[i, :8]]
            member_idxs = np.where(self.kmeans.labels_ == i)[0]
            examples = [reviews[idx] for idx in member_idxs[:3]]

            clusters.append(
                {
                    "cluster_id": i + 1,
                    "key_terms": top_terms,
                    "representative_examples": examples,
                    "size": int(len(member_idxs)),
                }
            )

        # Sort clusters by size descending
        clusters.sort(key=lambda c: c["size"], reverse=True)

        return {"total_reviews": len(reviews), "clusters": clusters}


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_reviews = [
        "The battery life is way too short, it barely lasts 4 hours.",
        "Really wish there was a dark mode option for the app.",
        "The screen flickers when the brightness is low.",
        "Charging takes forever, please support fast charging.",
        "The UI is a bit cluttered, hard to find the settings menu.",
        "Would love to see integration with Google Calendar.",
        "The device gets very hot during video calls.",
        "Awesome performance but the software needs a dark theme.",
        "Battery drains even when the device is idle.",
        "Please add a way to export data to CSV.",
        "Delivery was fast and packaging was excellent.",
        "The price is a bit high compared to competitors.",
    ]

    analyzer = FeedbackAnalyzer(n_clusters=3)
    result = analyzer.analyze_feedback(sample_reviews)

    for cluster in result["clusters"]:
        print(f"\nCluster {cluster['cluster_id']} ({cluster['size']} reviews)")
        print(f"  Key terms : {', '.join(cluster['key_terms'])}")
        print(f"  Example   : {cluster['representative_examples'][0]}")
