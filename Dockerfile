# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AI Review Guard (API + Training)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "\
import nltk; \
[nltk.download(p, quiet=True) for p in \
 ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"

# Copy source code
COPY . .

# Train model at build time if no pre-trained model exists
# (Remove this RUN if you want to supply pre-trained models separately)
RUN python train_model.py

# Default command — start the API
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
