"""
sentiment_analysis.py — VADER-based sentiment scoring.
Kept as a clean, typed, importable module.
"""

from typing import Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()  # Singleton — avoid rebuilding on every call


def analyze_sentiment(review_text: str) -> Dict[str, float]:
    """
    Score the emotional tone of a review using VADER.

    Returns:
        positive : fraction of positive sentiment
        neutral  : fraction of neutral sentiment
        negative : fraction of negative sentiment
        compound : normalised overall score [-1, +1]
        label    : "Positive" | "Neutral" | "Negative"
    """
    if not isinstance(review_text, str) or not review_text.strip():
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0, "compound": 0.0, "label": "Neutral"}

    scores = _analyzer.polarity_scores(review_text)
    compound = scores["compound"]
    label = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")

    return {
        "positive": round(scores["pos"], 4),
        "neutral": round(scores["neu"], 4),
        "negative": round(scores["neg"], 4),
        "compound": round(compound, 4),
        "label": label,
    }


if __name__ == "__main__":
    samples = [
        "This product is absolutely amazing! I love the quality.",
        "Terrible product. Broke after one day. Complete waste of money.",
        "It is okay. Nothing special, does the job I suppose.",
    ]
    for s in samples:
        r = analyze_sentiment(s)
        print(f"[{r['label']:8s}] {r['compound']:+.4f} | {s[:60]}")
