from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import uvicorn

# Import our custom modules
from predict import ReviewPredictor
from sentiment_analysis import analyze_sentiment
from feedback_analysis import FeedbackAnalyzer
from insight_engine import InsightEngine

app = FastAPI(
    title="AI Product Review API",
    description="API for detecting fake reviews and extracting product feedback insights.",
    version="2.0.0"
)

# Initialize components
try:
    predictor = ReviewPredictor()
    feedback_analyzer = FeedbackAnalyzer(n_clusters=3)
    insight_engine = InsightEngine()
except Exception as e:
    print(f"Warning: Initialization failed: {e}")
    # In a real app, you might want to handle this more gracefully
    predictor = None
    feedback_analyzer = None

# --- Data Models ---

class SingleReviewInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    reviews: List[str]

# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "AI Product Review Analysis API is running."
    }

@app.post("/predict-review")
async def predict_single_review(input_data: SingleReviewInput):
    """
    Predict if a single review is Fake or Genuine.
    Returns:
        - prediction (Fake/Genuine)
        - probability score
        - sentiment analysis (pos, neu, neg, compound)
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model predictor not initialized.")
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    try:
        # 1. Prediction & Probability
        prediction_result = predictor.predict(input_data.text)
        
        # 2. Sentiment Analysis
        sentiment_result = analyze_sentiment(input_data.text)
        
        return {
            "prediction": prediction_result["prediction"],
            "probability_score": prediction_result["confidence"],
            "sentiment": sentiment_result,
            "full_details": {
                "genuine_probability": prediction_result["genuine_probability"],
                "fake_probability": prediction_result["fake_probability"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-feedback")
async def analyze_batch_feedback(input_data: FeedbackInput):
    """
    Apply clustering and sentiment summary to a list of reviews.
    Returns:
        - topic clusters
        - sentiment summary
    """
    if not feedback_analyzer:
        raise HTTPException(status_code=503, detail="Feedback analyzer not initialized.")

    if not input_data.reviews:
        raise HTTPException(status_code=400, detail="Review list cannot be empty.")

    try:
        # Preprocess everything for classification
        cleaned_list = [predictor.preprocessor.clean_text(r) for r in input_data.reviews]
        
        # 1. Classification Filter (Detect Fake Reviews first)
        predictions = predictor.model.predict(predictor.vectorizer.transform(cleaned_list))
        genuine_reviews_orig = [input_data.reviews[i] for i, p in enumerate(predictions) if p == 1]
        
        # 2. Sequential Analysis (Only for Genuine)
        if not genuine_reviews_orig:
            return {
                "topic_clusters": [],
                "sentiment_summary": {"message": "No genuine reviews found for analysis."},
                "readable_insights": {"message": "All provided reviews were classified as potentially fake."}
            }

        # Cluster Analysis & Sentiment
        clustering_result = feedback_analyzer.analyze_feedback(genuine_reviews_orig)
        
        # Detailed Sentiment Analysis
        sentiment_scores = [analyze_sentiment(r) for r in genuine_reviews_orig]
        compounds = [s["compound"] for s in sentiment_scores]
        avg_sentiment = sum(compounds) / len(compounds)
        
        # Sentiment Distribution
        pos_count = sum(1 for s in compounds if s >= 0.05)
        neg_count = sum(1 for s in compounds if s <= -0.05)
        neu_count = len(genuine_reviews_orig) - pos_count - neg_count
        
        sentiment_label = "Positive" if avg_sentiment >= 0.05 else "Negative" if avg_sentiment <= -0.05 else "Neutral"
        
        # Human-readable Insights
        insights = insight_engine.generate_insights(genuine_reviews_orig)
        
        return {
            "topic_clusters": clustering_result.get("clusters", []),
            "sentiment_summary": {
                "average_compound_score": round(avg_sentiment, 4),
                "overall_sentiment": sentiment_label,
                "genuine_count": len(genuine_reviews_orig),
                "total_count": len(input_data.reviews),
                "negative_percentage": insights["summary_metrics"]["negative_review_percentage"],
                "distribution": {
                    "Positive": pos_count,
                    "Neutral": neu_count,
                    "Negative": neg_count
                }
            },
            "readable_insights": {
                "top_complaints": insights["top_complaints"],
                "feature_requests": insights["most_requested_features"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
