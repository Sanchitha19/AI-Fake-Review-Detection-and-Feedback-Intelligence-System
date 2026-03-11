"""
train_model.py — Production-grade training pipeline.

Features:
- Loads real data from reviews.csv (falls back to synthetic data if absent)
- Engineered behavioral features stacked with TF-IDF
- Ensemble classifier: Logistic Regression + Random Forest + XGBoost (soft voting)
- 5-fold stratified cross-validation
- SMOTE balancing when class imbalance is detected
- MLflow experiment tracking
- Saves model artefacts to models/
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from loguru import logger
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from preprocess import ReviewPreprocessor

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
DATA_PATH = Path("reviews.csv")
MLFLOW_EXPERIMENT = "fake-review-detection"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_data() -> pd.DataFrame:
    """Minimal synthetic fallback when reviews.csv is unavailable."""
    genuine = [
        "This product exceeded my expectations. High quality and fast shipping.",
        "I've been using this for a month and it works perfectly. Highly recommend.",
        "Great value for money. The build quality is solid and durable.",
        "Excellent customer service and the item arrived earlier than expected.",
        "Very happy with this purchase. It does exactly what it promises.",
        "The best version of this product I've owned. Well worth every penny.",
        "Sturdy design and very intuitive to use. Would buy again without hesitation.",
        "Beautifully packaged and high performing product. Impressed.",
        "Surprisingly good quality given the affordable price point.",
        "It's transformed my daily routine completely. Will definitely repurchase.",
        "Battery life is decent for the price. Solid build overall.",
        "Screen resolution is sharp and display is bright. Great for outdoor use.",
        "Setup was simple and fast. Works flawlessly out of the box.",
        "Camera quality is outstanding. Photos look professional.",
        "Sound quality from the speakers exceeded expectations.",
    ] * 15

    fake = [
        "BUY THIS NOW!!! AMAZING SCAM PRODUCT CHEAP!!!",
        "Worst purchase ever. Broke in one minute. FRAUD.",
        "Do not trust this seller. Waste of money and time.",
        "Absolutely garbage. Stay away. Fake descriptions everywhere.",
        "Cheap plastic feel. Not worth a single penny. Zero stars.",
        "Item never arrived. Customer service is completely nonexistent.",
        "Totally different from what was advertised. Massively disappointed.",
        "Zero stars. The product is a complete joke. Do not buy.",
        "Terrible experience. The item is literally falling apart immediately.",
        "SCAM ALERT! This is NOT the real product. Seller is fraudulent.",
        "BEST PRODUCT EVER!!! 5 STARS!!! BUY NOW!!! LIMITED TIME OFFER!!!",
        "I LOVE THIS SO MUCH!!!! PERFECT IN EVERY WAY!!!! AMAZING SELLER!!!!",
        "Must buy immediately! Best deal ever! Cannot believe this price!",
        "Seller paid me for review. Product is fine I guess.",
        "Everyone should buy this! Life-changing! Miracle product!!!",
    ] * 15

    df = pd.DataFrame(
        {
            "review_text": genuine + fake,
            "label": [1] * len(genuine) + [0] * len(fake),
        }
    )
    return df


def load_data() -> pd.DataFrame:
    """Load reviews.csv if present, otherwise fall back to synthetic data."""
    if DATA_PATH.exists():
        logger.info(f"Loading real dataset from {DATA_PATH}")

        # Use the Python csv engine so embedded commas inside quotes are handled
        # correctly; skip any structurally broken lines
        try:
            df = pd.read_csv(
                DATA_PATH,
                engine="python",
                on_bad_lines="skip",
                quoting=0,  # QUOTE_MINIMAL — honours quoted fields
            )
        except Exception as exc:
            logger.warning(f"CSV read failed ({exc}); falling back to synthetic data.")
            return _synthetic_data()

        # Normalise column names — support flexible schemas
        col_map: dict = {}
        for col in df.columns:
            if col.lower().strip() in ("review_text", "text", "review", "content"):
                col_map[col] = "review_text"
            elif col.lower().strip() in ("label", "fake", "is_fake", "class", "target"):
                col_map[col] = "label"
        df = df.rename(columns=col_map)

        if "review_text" not in df.columns or "label" not in df.columns:
            logger.warning(
                "reviews.csv does not have required columns; using synthetic data."
            )
            return _synthetic_data()

        df = df.dropna(subset=["review_text", "label"])

        # Map string labels → int  (genuine→1, fake→0)
        _label_map = {
            "genuine": 1, "real": 1, "1": 1, "true": 1,
            "fake": 0, "false": 0, "0": 0, "spam": 0,
        }
        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(_label_map)
        )

        bad = df["label"].isna().sum()
        if bad:
            logger.warning(f"Dropping {bad} rows with unrecognised label values.")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        logger.info(
            f"Loaded {len(df)} rows | "
            f"Genuine: {(df['label']==1).sum()} | Fake: {(df['label']==0).sum()}"
        )
        return df
    else:
        logger.warning("reviews.csv not found — using synthetic training data.")
        return _synthetic_data()


def check_balance(y: pd.Series) -> bool:
    """Return True if the dataset is severely imbalanced (minority < 35%)."""
    counts = y.value_counts(normalize=True)
    minority_ratio = counts.min()
    logger.info(f"Class distribution: {counts.to_dict()}")
    return minority_ratio < 0.35


def build_features(
    df: pd.DataFrame, preprocessor: ReviewPreprocessor, vectorizer: TfidfVectorizer, fit: bool = True
) -> csr_matrix:
    """
    Combine TF-IDF text features + behavioural numeric features into a single
    sparse matrix for model training.
    """
    # Text features
    cleaned = df["review_text"].apply(preprocessor.clean_text)
    if fit:
        text_feats = vectorizer.fit_transform(cleaned)
    else:
        text_feats = vectorizer.transform(cleaned)

    # Behavioural features (numeric, sparse-compatible)
    feat_cols = [
        "review_length", "word_count", "exclamation_count",
        "caps_ratio", "avg_word_length", "punctuation_density", "digit_ratio",
    ]
    df_features = df.copy()
    df_features = preprocessor.process_dataframe(df_features)

    numeric_feats = csr_matrix(df_features[feat_cols].fillna(0).values)
    return hstack([text_feats, numeric_feats])


def train_and_evaluate() -> None:
    """Main training entrypoint with MLflow tracking."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="ensemble_tfidf_behavioral"):
        # ── 1. Data ──────────────────────────────────────────────────────────
        df = load_data()
        preprocessor = ReviewPreprocessor()
        vectorizer = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2), sublinear_tf=True
        )

        X_all = build_features(df, preprocessor, vectorizer, fit=True)
        y_all = df["label"].values

        # ── 2. Balance Check ─────────────────────────────────────────────────
        needs_smote = check_balance(pd.Series(y_all))
        if needs_smote:
            logger.info("Applying SMOTE to balance classes…")
            sm = SMOTE(random_state=42)
            X_all, y_all = sm.fit_resample(X_all, y_all)
            logger.info(
                f"After SMOTE — total: {len(y_all)} | "
                f"Genuine: {(y_all==1).sum()} | Fake: {(y_all==0).sum()}"
            )

        mlflow.log_param("n_samples", len(y_all))
        mlflow.log_param("smote_applied", needs_smote)
        mlflow.log_param("tfidf_max_features", 8000)

        # ── 3. Train/Test Split ───────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        # ── 4. Ensemble Model ─────────────────────────────────────────────────
        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )

        ensemble = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
            voting="soft",
            weights=[1, 1, 1],
        )

        # ── 5. Cross-Validation ───────────────────────────────────────────────
        logger.info("Running 5-fold stratified cross-validation…")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(ensemble, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1)
        logger.info(f"CV F1 scores: {cv_f1.round(4)} | Mean: {cv_f1.mean():.4f}")
        mlflow.log_metric("cv_f1_mean", float(cv_f1.mean()))
        mlflow.log_metric("cv_f1_std", float(cv_f1.std()))

        # ── 6. Final Fit & Evaluation ─────────────────────────────────────────
        logger.info("Training final ensemble on full training split…")
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        logger.info(
            f"\n{'='*50}\nTest Set Results:\n"
            f"  Accuracy : {acc:.4f}\n"
            f"  Precision: {prec:.4f}\n"
            f"  Recall   : {rec:.4f}\n"
            f"  F1 Score : {f1:.4f}\n"
            f"{'='*50}"
        )
        logger.info("\n" + classification_report(y_test, y_pred, target_names=["Fake", "Genuine"]))

        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

        # ── 7. Save Artefacts ─────────────────────────────────────────────────
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "best_classifier.joblib"
        vec_path   = MODELS_DIR / "vectorizer.joblib"
        prep_path  = MODELS_DIR / "preprocessor.joblib"

        joblib.dump(ensemble,     model_path)
        joblib.dump(vectorizer,   vec_path)
        joblib.dump(preprocessor, prep_path)

        mlflow.sklearn.log_model(ensemble, "ensemble_model")
        mlflow.log_artifact(str(vec_path))

        logger.info(f"Model saved → {model_path}")
        logger.info(f"Vectorizer saved → {vec_path}")
        logger.info("✅ Training pipeline complete!")


if __name__ == "__main__":
    train_and_evaluate()
