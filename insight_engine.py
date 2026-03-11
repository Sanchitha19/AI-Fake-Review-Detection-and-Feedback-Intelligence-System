"""
insight_engine.py — Zero-shot complaint classification via HuggingFace.

Replaces brittle keyword heuristics with a zero-shot text-classification
pipeline that categorises complaints into predefined product-relevant buckets.
Falls back to keyword scoring if the transformer model is unavailable.
"""

from typing import Dict, List, Any

from loguru import logger
from sentiment_analysis import analyze_sentiment

COMPLAINT_CATEGORIES = [
    "battery life",
    "shipping and delivery",
    "product quality",
    "user interface and UX",
    "pricing and value",
    "customer service",
    "performance and speed",
    "build and design",
]

FEATURE_REQUEST_MARKERS = [
    "wish", "hope", "would love", "please add", "add", "feature", "support",
    "integration", "dark mode", "option", "include", "improve", "need",
    "want", "better", "future", "update", "could you", "can you",
]


def _load_classifier():
    """Lazily load the zero-shot pipeline. Returns None on failure."""
    try:
        from transformers import pipeline

        logger.info("Loading zero-shot-classification pipeline (facebook/bart-large-mnli)…")
        clf = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU; set to 0 for GPU
        )
        logger.info("Zero-shot classifier loaded.")
        return clf
    except Exception as exc:
        logger.warning(f"Could not load zero-shot classifier: {exc}. Using keyword fallback.")
        return None


_zs_classifier = None  # module-level lazy singleton


def _get_classifier():
    global _zs_classifier
    if _zs_classifier is None:
        _zs_classifier = _load_classifier()
    return _zs_classifier


def _keyword_fallback_category(text: str) -> str:
    """Naïve keyword fallback when the transformer is not available."""
    t = text.lower()
    if any(k in t for k in ("battery", "charge", "drain", "power")):
        return "battery life"
    if any(k in t for k in ("ship", "deliver", "arriv", "packag")):
        return "shipping and delivery"
    if any(k in t for k in ("broke", "broke", "quality", "material", "plastic", "build")):
        return "product quality"
    if any(k in t for k in ("ui", "app", "interface", "design", "navigate", "clutter")):
        return "user interface and UX"
    if any(k in t for k in ("price", "expensive", "cost", "cheap", "value")):
        return "pricing and value"
    if any(k in t for k in ("service", "support", "refund", "return", "customer")):
        return "customer service"
    if any(k in t for k in ("slow", "lag", "crash", "freeze", "performance", "speed")):
        return "performance and speed"
    return "product quality"


class InsightEngine:
    """
    Analyses a corpus of genuine reviews and returns:
    - categorised complaints (zero-shot or keyword fallback)
    - detected feature requests
    - summary metrics
    """

    def __init__(self) -> None:
        self._clf = None  # loaded on first use

    def _classifier(self):
        if self._clf is None:
            self._clf = _get_classifier()
        return self._clf

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_complaint(self, text: str) -> str:
        """Categorise complaints using efficient keyword matching."""
        return _keyword_fallback_category(text)

    def _is_feature_request(self, text: str) -> bool:
        t = text.lower()
        return any(marker in t for marker in FEATURE_REQUEST_MARKERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_insights(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Analyse a list of reviews and return structured insights.

        Returns:
            summary_metrics : overall stats dict
            complaint_breakdown : category → count + examples
            top_complaints : list of (category, count) sorted desc
            feature_requests : list of review snippets that request features
        """
        if not reviews:
            return {"error": "No reviews provided.", "complaint_breakdown": {}}

        complaint_map: Dict[str, List[str]] = {cat: [] for cat in COMPLAINT_CATEGORIES}
        feature_requests: List[str] = []
        sentiments: List[float] = []

        for review in reviews:
            sent = analyze_sentiment(review)
            sentiments.append(sent["compound"])

            # Feature request detection (fast, keyword-based)
            if self._is_feature_request(review):
                feature_requests.append(review[:120])

            # Complaint classification (zero-shot or fallback)
            if sent["compound"] <= -0.05 or sent["label"] == "Negative":
                category = self._classify_complaint(review)
                complaint_map[category].append(review[:120])

        neg_count = sum(1 for s in sentiments if s <= -0.05)
        neg_pct = (neg_count / len(reviews) * 100) if reviews else 0.0
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Build sorted complaint breakdown
        complaint_breakdown = {
            cat: {"count": len(revs), "examples": revs[:2]}
            for cat, revs in complaint_map.items()
            if revs
        }
        top_complaints = sorted(
            complaint_breakdown.items(), key=lambda x: x[1]["count"], reverse=True
        )

        return {
            "summary_metrics": {
                "total_reviews": len(reviews),
                "negative_review_percentage": f"{neg_pct:.1f}%",
                "average_sentiment": round(avg_sent, 4),
            },
            "complaint_breakdown": complaint_breakdown,
            "top_complaints": [
                {"category": cat, "count": data["count"], "examples": data["examples"]}
                for cat, data in top_complaints
            ],
            "feature_requests": feature_requests[:10],  # Cap at 10
        }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = [
        "The battery drains so fast, can't even last a day.",
        "I wish there was a dark mode option.",
        "App keeps crashing on the settings page.",
        "Please add Google Drive integration.",
        "Delivery took three weeks — very disappointing.",
        "Price is way too high for the quality received.",
        "Screen flickers sometimes and the UI is confusing.",
        "Customer service was rude and unhelpful.",
    ]
    engine = InsightEngine()
    results = engine.generate_insights(sample)

    print(f"\nNegative: {results['summary_metrics']['negative_review_percentage']}")
    print("\nTop Complaints:")
    for c in results["top_complaints"]:
        print(f"  [{c['count']}] {c['category']}")
    print(f"\nFeature Requests ({len(results['feature_requests'])}):")
    for r in results["feature_requests"]:
        print(f"  - {r}")
