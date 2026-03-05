import joblib
import os
import numpy as np
from preprocess import ReviewPreprocessor

class ReviewPredictor:
    def __init__(self, model_path='models/best_classifier.joblib', vec_path='models/vectorizer.joblib'):
        self.model_path = model_path
        self.vec_path = vec_path
        self.preprocessor = ReviewPreprocessor()
        
        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
        else:
            raise FileNotFoundError("Model or Vectorizer not found. Please run train_model.py first.")

    def predict(self, review_text):
        """
        Predict if a review is Fake or Genuine and return the probability.
        """
        # 1. Preprocess
        cleaned_text = self.preprocessor.clean_text(review_text)
        
        # 2. Vectorize
        features = self.vectorizer.transform([cleaned_text])
        
        # 3. Predict
        prediction = self.model.predict(features)[0]
        
        # 4. Get Probability (if supported by model)
        try:
            proba = self.model.predict_proba(features)[0]
            # Assuming classes are [0: Fake, 1: Genuine]
            # The indices depend on how the model was trained, usually sorted alphabetically or numerically
            # For 0/1 labels, proba[1] is Genuine, proba[0] is Fake
            genuine_prob = proba[1]
            fake_prob = proba[0]
        except AttributeError:
            genuine_prob = 1.0 if prediction == 1 else 0.0
            fake_prob = 1.0 if prediction == 0 else 0.0
            
        result = "Genuine" if prediction == 1 else "Fake"
        confidence = genuine_prob if prediction == 1 else fake_prob
        
        return {
            "prediction": result,
            "confidence": float(confidence),
            "genuine_probability": float(genuine_prob),
            "fake_probability": float(fake_prob)
        }

if __name__ == "__main__":
    import sys
    
    predictor = ReviewPredictor()
    
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        # Default test case if no argument provided
        input_text = "This is a wonderful product, I love the quality and the speed of delivery!"
        print(f"No input provided. Using default test: '{input_text}'")
    
    res = predictor.predict(input_text)
    
    print("\n--- Prediction Result ---")
    print(f"Input: {input_text}")
    print(f"Result: {res['prediction']}")
    print(f"Confidence: {res['confidence']:.2%}")
    print(f"Full Probabilities: Fake={res['fake_probability']:.2%}, Genuine={res['genuine_probability']:.2%}")
