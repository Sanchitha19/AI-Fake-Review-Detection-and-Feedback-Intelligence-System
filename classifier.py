import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from preprocess import ReviewPreprocessor

class FakeReviewClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def train(self, reviews, labels):
        """
        Train the model using provided reviews and labels.
        Labels: 0 for Fake, 1 for Genuine (or vice versa depending on dataset)
        """
        X = self.vectorizer.fit_transform(reviews)
        y = labels
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Test performance
        y_pred = self.model.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def predict(self, reviews):
        """
        Predict if reviews are fake or genuine.
        """
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        X = self.vectorizer.transform(reviews)
        return self.model.predict(X)

    def save_model(self, model_path='models/classifier.pkl', vec_path='models/vectorizer.pkl'):
        """
        Save the model and vectorizer to disk.
        """
        import os
        os.makedirs('models', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vec_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self, model_path='models/classifier.pkl', vec_path='models/vectorizer.pkl'):
        """
        Load the model and vectorizer from disk.
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vec_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_trained = True

if __name__ == "__main__":
    # Synthetic dataset for demonstration
    data = {
        'review_text': [
            "Great product, highly recommend!", "Love this item, works perfectly.",
            "Amazing quality, fast shipping.", "Best purchase I've made this year.",
            "Buy this now! You won't regret it.", "Excellent service and product.",
            "Cheap plastic, broke immediately.", "Total waste of money.",
            "Don't buy this, it's a scam.", "Awful experience, never again.",
            "Does not work as advertised.", "Poor quality and customer service."
        ] * 10,
        'label': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 10 # 1: Genuine, 0: Fake (Simplified)
    }
    
    df = pd.DataFrame(data)
    preprocessor = ReviewPreprocessor()
    df['cleaned'] = df['review_text'].apply(preprocessor.clean_text)
    
    classifier = FakeReviewClassifier()
    classifier.train(df['cleaned'], df['label'])
    classifier.save_model()
    
    # Test prediction
    test_reviews = ["This is the best thing ever!", "Scam alert, do not buy."]
    test_cleaned = [preprocessor.clean_text(r) for r in test_reviews]
    predictions = classifier.predict(test_cleaned)
    print(f"\nPredictions: {predictions} (1: Genuine, 0: Fake)")
