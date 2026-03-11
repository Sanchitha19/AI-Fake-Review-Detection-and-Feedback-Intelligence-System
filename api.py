"""
api.py — Production FastAPI backend.

Endpoints:
  GET  /              — root health ping
  GET  /health        — detailed health check
  GET  /metrics       — runtime metrics
  POST /predict-review — single review classification + sentiment
  POST /analyze-feedback — batch analysis (clustering + insights)
  POST /batch-predict  — CSV upload for bulk predictions
  GET  /job/{job_id}   — poll status of a background batch job

Security:
  Bearer token auth on all non-health endpoints (set API_SECRET_KEY in .env).
  Pass Authorization: Bearer <key> in request headers.
"""

import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env if present
load_dotenv()

API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "dev-secret-key-change-me")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# ---------------------------------------------------------------------------
# Lazy imports for heavy ML modules (avoid import-time crashes)
# ---------------------------------------------------------------------------
_predictor = None
_feedback_analyzer = None
_insight_engine = None
_startup_time = time.time()
_request_count = 0


def _get_predictor():
    global _predictor
    if _predictor is None:
        from predict import ReviewPredictor
        _predictor = ReviewPredictor()
    return _predictor


def _get_feedback_analyzer():
    global _feedback_analyzer
    if _feedback_analyzer is None:
        from feedback_analysis import FeedbackAnalyzer
        _feedback_analyzer = FeedbackAnalyzer(n_clusters=4)
    return _feedback_analyzer


def _get_insight_engine():
    global _insight_engine
    if _insight_engine is None:
        from insight_engine import InsightEngine
        _insight_engine = InsightEngine()
    return _insight_engine


# ---------------------------------------------------------------------------
# Background Job Store (in-memory; swap for Redis in production)
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = {}  # job_id → {"status", "result", "created_at"}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API starting up — pre-loading ML models…")
    try:
        _get_predictor()
        _get_feedback_analyzer()
        logger.info("✅ ML models loaded.")
    except Exception as exc:
        logger.warning(f"Model pre-load warning: {exc}")
    yield
    logger.info("🛑 API shutting down.")


app = FastAPI(
    title="AI Product Review Guard API",
    description=(
        "Detect fake reviews, analyse customer sentiment, "
        "cluster feedback themes, and extract actionable product insights."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_bearer = HTTPBearer(auto_error=False)


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    if API_SECRET_KEY == "dev-secret-key-change-me":
        return "dev"  # Skip auth in dev mode
    if credentials is None or credentials.credentials != API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED", "message": "Invalid or missing Bearer token."},
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SingleReviewInput(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000, description="The review text to analyse.")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class FeedbackInput(BaseModel):
    reviews: List[str] = Field(..., min_length=1, max_length=500, description="List of review texts.")

    @field_validator("reviews")
    @classmethod
    def validate_reviews(cls, v: List[str]) -> List[str]:
        cleaned = [r.strip() for r in v if r.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty review is required.")
        return cleaned


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    genuine_probability: float
    fake_probability: float
    sentiment: Dict[str, Any]
    behavioral_signals: Dict[str, float]


class ErrorResponse(BaseModel):
    error_code: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "version": app.version, "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check — no auth required."""
    uptime = round(time.time() - _startup_time, 1)
    model_ok = Path("models/best_classifier.joblib").exists()
    return {
        "status": "healthy" if model_ok else "degraded",
        "uptime_seconds": uptime,
        "model_ready": model_ok,
        "version": app.version,
    }


@app.get("/metrics", tags=["Health"], dependencies=[Depends(verify_token)])
async def metrics():
    """Runtime stats — requires auth."""
    global _request_count
    return {
        "total_requests_served": _request_count,
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "active_jobs": len(_jobs),
    }


@app.post(
    "/predict-review",
    response_model=PredictionResponse,
    tags=["Prediction"],
    dependencies=[Depends(verify_token)],
)
async def predict_single_review(body: SingleReviewInput):
    """
    Predict whether a single review is **Fake** or **Genuine**.

    Returns prediction confidence, full probability breakdown,
    VADER sentiment scores, and behavioral engineering signals.
    """
    global _request_count
    _request_count += 1

    try:
        predictor = _get_predictor()
        pred = predictor.predict(body.text)

        from sentiment_analysis import analyze_sentiment
        sentiment = analyze_sentiment(body.text)

        return {
            "prediction": pred["prediction"],
            "confidence": pred["confidence"],
            "genuine_probability": pred["genuine_probability"],
            "fake_probability": pred["fake_probability"],
            "sentiment": sentiment,
            "behavioral_signals": pred["behavioral_signals"],
        }
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error_code": "MODEL_NOT_FOUND", "message": str(exc)},
        )
    except Exception as exc:
        logger.exception(f"predict-review error: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "INTERNAL_ERROR", "message": str(exc)},
        )


@app.post("/analyze-feedback", tags=["Analysis"], dependencies=[Depends(verify_token)])
async def analyze_batch_feedback(body: FeedbackInput):
    """
    Analyse a batch of reviews:
    1. Filter out fake reviews.
    2. Cluster genuine reviews into themes.
    3. Classify complaints into categories.
    4. Compute sentiment distribution.
    """
    global _request_count
    _request_count += 1

    try:
        predictor = _get_predictor()
        feedback_analyzer = _get_feedback_analyzer()
        insight_engine = _get_insight_engine()

        from sentiment_analysis import analyze_sentiment

        # Step 1 — classify each review
        results = predictor.predict_batch(body.reviews)
        genuine_reviews = [
            r for r, res in zip(body.reviews, results) if res["prediction"] == "Genuine"
        ]

        if not genuine_reviews:
            return {
                "total_submitted": len(body.reviews),
                "genuine_count": 0,
                "fake_count": len(body.reviews),
                "topic_clusters": [],
                "sentiment_summary": {"message": "All reviews classified as Fake."},
                "insights": {"message": "No genuine reviews to analyse."},
            }

        # Step 2 — sentiment distribution
        sentiments = [analyze_sentiment(r) for r in genuine_reviews]
        compounds = [s["compound"] for s in sentiments]
        avg_sent = sum(compounds) / len(compounds)
        pos_n = sum(1 for c in compounds if c >= 0.05)
        neg_n = sum(1 for c in compounds if c <= -0.05)
        neu_n = len(genuine_reviews) - pos_n - neg_n

        # Step 3 — clustering
        clustering = feedback_analyzer.analyze_feedback(genuine_reviews)

        # Step 4 — insights
        insights = insight_engine.generate_insights(genuine_reviews)

        return {
            "total_submitted": len(body.reviews),
            "genuine_count": len(genuine_reviews),
            "fake_count": len(body.reviews) - len(genuine_reviews),
            "topic_clusters": clustering.get("clusters", []),
            "sentiment_summary": {
                "average_compound": round(avg_sent, 4),
                "overall_label": "Positive" if avg_sent >= 0.05 else ("Negative" if avg_sent <= -0.05 else "Neutral"),
                "distribution": {"Positive": pos_n, "Neutral": neu_n, "Negative": neg_n},
            },
            "insights": {
                "summary_metrics": insights.get("summary_metrics", {}),
                "top_complaints": insights.get("top_complaints", []),
                "feature_requests": insights.get("feature_requests", []),
            },
        }

    except Exception as exc:
        logger.exception(f"analyze-feedback error: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "INTERNAL_ERROR", "message": str(exc)},
        )


def _run_batch_job(job_id: str, reviews: List[str]) -> None:
    """Background worker for CSV batch prediction jobs."""
    _jobs[job_id]["status"] = "running"
    try:
        predictor = _get_predictor()
        from sentiment_analysis import analyze_sentiment

        # 1. Batch predict authenticity (Vectorized)
        logger.info(f"Starting batch prediction for job {job_id} ({len(reviews)} reviews)")
        predictions = predictor.predict_batch(reviews)

        # 2. Add sentiment analysis
        output_rows = []
        for rev, pred in zip(reviews, predictions):
            sent = analyze_sentiment(rev)
            output_rows.append(
                {
                    "review_text": rev,
                    "prediction": pred["prediction"],
                    "confidence": pred["confidence"],
                    "genuine_probability": pred["genuine_probability"],
                    "fake_probability": pred["fake_probability"],
                    "sentiment_label": sent["label"],
                    "sentiment_compound": sent["compound"],
                }
            )

        _jobs[job_id]["result"] = output_rows
        _jobs[job_id]["status"] = "done"
        logger.info(f"Batch job {job_id} complete — {len(output_rows)} rows.")
    except Exception as exc:
        logger.exception(f"Batch job {job_id} failed: {exc}")
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


@app.post("/batch-predict", tags=["Prediction"], dependencies=[Depends(verify_token)])
async def batch_predict_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with a 'review_text' column."),
):
    """
    Upload a CSV file for async bulk prediction.
    Returns a `job_id` — poll `/job/{job_id}` for status and results.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "INVALID_FILE", "message": "Only CSV files are accepted."},
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "PARSE_ERROR", "message": "Could not parse CSV file."},
        )

    col = next((c for c in df.columns if c.lower() in ("review_text", "text", "review")), None)
    if col is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "MISSING_COLUMN",
                "message": "CSV must contain a 'review_text' column.",
            },
        )

    reviews = df[col].dropna().astype(str).tolist()
    if not reviews:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "EMPTY_DATA", "message": "CSV has no valid review rows."},
        )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "created_at": time.time(), "total": len(reviews)}
    background_tasks.add_task(_run_batch_job, job_id, reviews)

    return {"job_id": job_id, "status": "queued", "total_reviews": len(reviews)}


@app.get("/job/{job_id}", tags=["Prediction"], dependencies=[Depends(verify_token)])
async def get_job_status(job_id: str):
    """Poll the status of a background batch prediction job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "JOB_NOT_FOUND", "message": f"No job with id '{job_id}'."},
        )
    return {
        "job_id": job_id,
        "status": job["status"],
        "total_reviews": job.get("total"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT, reload=True)
