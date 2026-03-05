import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from preprocess import ReviewPreprocessor

def create_synthetic_data():
    """
    Creates a larger synthetic dataset for training demonstration.
    1: Genuine, 0: Fake
    """
    genuine_reviews = [
        "This product exceeded my expectations. High quality and fast shipping.",
        "I've been using this for a month and it works perfectly. Highly recommend.",
        "Great value for money. The build quality is solid.",
        "Excellent customer service and the item arrived earlier than expected.",
        "Very happy with this purchase. It does exactly what it says.",
        "The best version of this product I've owned. Five stars!",
        "Sturdy design and very intuitive to use. Well worth it.",
        "Beautifully packaged and high performing product.",
        "Surprisingly good quality given the price point.",
        "I would definitely buy this again. It's transformed my daily routine."
    ] * 20
    
    fake_reviews = [
        "BUY THIS NOW!!! AMAZING SCAM PRODUCT CHEAP!!!",
        "Worst purchase ever. Broke in one minute. FRAUD.",
        "Do not trust this seller. Waste of money and time.",
        "Absolutely garbage. Stay away. Fake descriptions.",
        "Cheap plastic feel. Not worth a single penny.",
        "Item never arrived. Customer service is nonexistent.",
        "Totally different from what was advertised. Disappointed.",
        "Zero stars. The product is a complete joke.",
        "Terrible experience. The item is literally falling apart.",
        "Scam alert! This is not the real product."
    ] * 20
    
    data = {
        'review_text': genuine_reviews + fake_reviews,
        'label': [1] * len(genuine_reviews) + [0] * len(fake_reviews)
    }
    return pd.DataFrame(data)

def train_and_evaluate_models():
    # 1. Load/Create dataset
    print("Loading/Creating dataset...")
    df = create_synthetic_data()
    
    # 2. Preprocess
    print("Preprocessing text...")
    preprocessor = ReviewPreprocessor()
    df['cleaned_review'] = df['review_text'].apply(preprocessor.clean_text)
    
    # 3. Feature Extraction
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_f1 = -1
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }
        
        print(f"{name} Results:")
        print(f" - Accuracy:  {acc:.4f}")
        print(f" - Precision: {prec:.4f}")
        print(f" - Recall:    {rec:.4f}")
        print(f" - F1 Score:  {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    # 5. Save the best model and vectorizer
    print(f"\nBest Model: {best_name} (F1: {best_f1:.4f})")
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/best_classifier.joblib'
    vec_path = 'models/vectorizer.joblib'
    
    print(f"Saving best model to {model_path}...")
    joblib.dump(best_model, model_path)
    
    print(f"Saving vectorizer to {vec_path}...")
    joblib.dump(vectorizer, vec_path)
    
    print("\nTraining completed successfully.")

if __name__ == "__main__":
    train_and_evaluate_models()
