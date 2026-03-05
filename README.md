# AI Fake Review Detector & Insight Extractor

This project detects fraudulent product reviews and extracts product insights from genuine customer feedback using Machine Learning and NLP.

## Features
- **Machine Learning Classifier**: Detects patterns indicative of fake or genuine reviews.
- **NLP Preprocessing**: Lemmatization, stopword removal, and noise reduction.
- **Sentiment Analysis**: Tracks customer emotional tone (VADER + TextBlob).
- **Topic Clustering**: Groups genuine feedback into recurring themes for actionable insights.
- **FastAPI Backend**: Professional RESTful API for high-performance analysis.
- **Streamlit Dashboard**: Interactive UI for review visualization and summary metrics.

## Tech Stack
- **Python**: Core logic
- **scikit-learn**: Classification and Clustering
- **FastAPI / Uvicorn**: Backend
- **Streamlit**: Web Dashboard
- **NLTK**: NLP preprocessing
- **Matplotlib / Seaborn**: Visualizations

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the initial model (Synthetic data demo):
   ```bash
   python classifier.py
   ```

## Usage

1. Start the FastAPI backend:
   ```bash
   uvicorn api:app --reload
   ```

2. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Folder Structure
```text
project/
├── app.py           # Streamlit dashboard
├── api.py           # FastAPI backend
├── preprocess.py    # NLP processing pipeline
├── classifier.py    # Fake review ML model
├── sentiment.py     # Sentiment analysis module
├── clustering.py    # Topic modeling/clustering
├── requirements.txt # Project dependencies
└── models/          # Saved ML models/vectorizers
```
