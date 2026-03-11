"""
predict.py — Inference wrapper that loads the trained ensemble and returns
predictions enriched with confidence scores and behavioral signals.
"""

import os
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from loguru import logger
from scipy.sparse import hstack, csr_matrix

from preprocess import ReviewPreprocessor

MODELS_DIR = Path("models")


class ReviewPredictor:
    """
    Loads the saved ensemble model + TF-IDF vectorizer and predicts
    whether a review is Fake or Genuine with full probability breakdown.
    """

    def __init__(
        self,
        model_path: str = "models/best_classifier.joblib",
        vec_path: str = "models/vectorizer.joblib",
    ) -> None:
        model_p = Path(model_path)
        vec_p = Path(vec_path)

        if not model_p.exists() or not vec_p.exists():
            raise FileNotFoundError(
                "Model or vectorizer not found. Run `python train_model.py` first."
            )

        self.model = joblib.load(model_p)
        self.vectorizer = joblib.load(vec_p)
        self.preprocessor = ReviewPreprocessor()
        logger.info("ReviewPredictor loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_feature_matrix(self, reviews: list[str]) -> csr_matrix:
        """Build feature matrix for a batch of reviews efficiently."""
        import pandas as pd
        
        # 1. Clean all text
        cleaned_list = [self.preprocessor.clean_text(r) for r in reviews]
        text_feats = self.vectorizer.transform(cleaned_list)

        # 2. Extract behavioral features for all
        # process_dataframe is already vectorized for behavioral features
        df_temp = pd.DataFrame({"review_text": reviews})
        df_temp = self.preprocessor.process_dataframe(df_temp, text_column="review_text")
        
        feat_cols = [
            "review_length", "word_count", "exclamation_count",
            "caps_ratio", "avg_word_length", "punctuation_density", "digit_ratio",
        ]
        numeric_feats = csr_matrix(df_temp[feat_cols].values)
        
        return hstack([text_feats, numeric_feats])

    def predict(self, review_text: str) -> Dict[str, Any]:
        """
        Predict authenticity of a single review.

        Returns:
            prediction          : "Genuine" | "Fake"
            confidence          : probability of the predicted class
            genuine_probability : P(Genuine)
            fake_probability    : P(Fake)
            behavioral_signals  : dict of engineered feature values
        """
        if not review_text or not review_text.strip():
            raise ValueError("review_text cannot be empty.")

        features = self._build_feature_matrix([review_text])
        pred_class = self.model.predict(features)[0]

        try:
            proba = self.model.predict_proba(features)[0]
            genuine_prob = float(proba[1])
            fake_prob = float(proba[0])
        except AttributeError:
            genuine_prob = 1.0 if pred_class == 1 else 0.0
            fake_prob = 1.0 if pred_class == 0 else 0.0

        label = "Genuine" if pred_class == 1 else "Fake"
        confidence = genuine_prob if pred_class == 1 else fake_prob

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "genuine_probability": round(genuine_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "behavioral_signals": self.preprocessor.extract_behavioral_features(review_text),
        }

    def predict_batch(self, reviews: list[str]) -> list[Dict[str, Any]]:
        """Run batch prediction efficiently using matrix operations."""
        if not reviews:
            return []

        features = self._build_feature_matrix(reviews)
        preds = self.model.predict(features)
        
        try:
            probas = self.model.predict_proba(features)
        except AttributeError:
            # Fallback if model doesn't support predict_proba
            probas = [[0.0, 1.0] if p == 1 else [1.0, 0.0] for p in preds]

        results = []
        for i, (text, pred_class) in enumerate(zip(reviews, preds)):
            genuine_prob = float(probas[i][1])
            fake_prob = float(probas[i][0])
            label = "Genuine" if pred_class == 1 else "Fake"
            confidence = genuine_prob if pred_class == 1 else fake_prob
            
            # Behavioral signals for return (already computed in matrix, but we need the dict for individual results)
            # Re-running extraction for the dict is small compared to model overhead
            results.append({
                "prediction": label,
                "confidence": round(confidence, 4),
                "genuine_probability": round(genuine_prob, 4),
                "fake_probability": round(fake_prob, 4),
                "behavioral_signals": self.preprocessor.extract_behavioral_features(text),
            })
        
        return results


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    predictor = ReviewPredictor()
    text = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "This is a wonderful product — love the quality and fast delivery!"
    )

    res = predictor.predict(text)
    print(f"\nInput  : {text}")
    print(f"Result : {res['prediction']}  (confidence {res['confidence']:.2%})")
    print(f"P(Genuine)={res['genuine_probability']:.4f}  P(Fake)={res['fake_probability']:.4f}")
    print(f"Signals: {res['behavioral_signals']}")
