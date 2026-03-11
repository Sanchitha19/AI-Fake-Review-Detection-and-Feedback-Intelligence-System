"""
app.py — Production-grade Streamlit dashboard for the AI Review Guard system.

Layout:
  Tab 1 — Single Review Check     (live prediction + sentiment gauge)
  Tab 2 — Batch Analysis          (CSV upload, progress bar, color-coded table)
  Tab 3 — Insights Dashboard      (complaint breakdown, cluster explorer, PDFs)

Requirements: streamlit, plotly, fpdf2, requests, pandas, python-dotenv
"""

import io
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# ── env ──────────────────────────────────────────────────────────────────────
load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_SECRET_KEY", "dev-secret-key-change-me")
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Review Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark background */
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(22,27,34,0.9);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }

    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #8b949e;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(46,160,67,0.4);
    }

    /* Alerts */
    .genuine-badge {
        background: linear-gradient(135deg, #0f2a18, #1a4a2e);
        border: 1px solid #2ea043;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .fake-badge {
        background: linear-gradient(135deg, #2a0f0f, #4a1a1a);
        border: 1px solid #f85149;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* Divider */
    hr { border-color: #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _api_ok() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200 and r.json().get("model_ready", False)
    except Exception:
        return False


def _predict(text: str) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{API_BASE}/predict-review",
            json={"text": text},
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
    except requests.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure `python api.py` is running.")
    return None


def _analyze(reviews: List[str]) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{API_BASE}/analyze-feedback",
            json={"reviews": reviews},
            headers=HEADERS,
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
    except requests.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure `python api.py` is running.")
    return None


def _batch_predict_csv(file_bytes: bytes) -> Optional[str]:
    """Submit CSV for background batch job and poll until complete."""
    try:
        resp = requests.post(
            f"{API_BASE}/batch-predict",
            files={"file": ("upload.csv", file_bytes, "text/csv")},
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code != 200:
            st.error(f"Upload error: {resp.json().get('detail', resp.text)}")
            return None

        job_id = resp.json()["job_id"]
        total = resp.json().get("total_reviews", "?")
        progress_bar = st.progress(0, text=f"Processing {total} reviews…")
        elapsed = 0

        while elapsed < 300:
            time.sleep(2)
            elapsed += 2
            poll = requests.get(f"{API_BASE}/job/{job_id}", headers=HEADERS, timeout=10)
            info = poll.json()
            if info["status"] == "done":
                progress_bar.progress(1.0, text="✅ Done!")
                return info["result"]
            elif info["status"] == "error":
                st.error(f"Batch job failed: {info.get('error')}")
                return None
            pct = min(elapsed / 60, 0.95)
            progress_bar.progress(pct, text=f"Running… ({elapsed}s elapsed)")

        st.warning("Job is taking too long. Check the API logs.")
        return None
    except requests.ConnectionError:
        st.error("❌ Cannot reach the API.")
        return None


def _make_pdf(df: pd.DataFrame, summary: Dict) -> bytes:
    """Generate a simple PDF report from batch results."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 130, 76)
    # Use standard hyphen instead of em-dash to avoid encoding errors
    pdf.cell(0, 12, "AI Review Guard - Batch Report", ln=True)

    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, f"Total Reviews: {summary.get('total', len(df))}", ln=True)
    pdf.cell(0, 8, f"Genuine: {summary.get('genuine', 0)}   Fake: {summary.get('fake', 0)}", ln=True)
    pdf.cell(0, 8, f"Overall Sentiment: {summary.get('sentiment', 'N/A')}", ln=True)
    pdf.ln(6)

    # Table header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 240, 220)
    col_widths = [100, 24, 24, 28, 28]
    headers = ["Review (truncated)", "Pred.", "Conf.", "Genuine P", "Fake P"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", size=9)
    for _, row in df.head(50).iterrows():
        # Sanitize text to remove non-latin1 characters for basic FPDF fonts
        raw_text = str(row.get("review_text", ""))[:55]
        clean_text = raw_text.encode("latin-1", "ignore").decode("latin-1")
        
        pdf.cell(col_widths[0], 7, clean_text, border=1)
        pdf.cell(col_widths[1], 7, str(row.get("prediction", "")), border=1)
        pdf.cell(col_widths[2], 7, f"{row.get('confidence', 0):.0%}", border=1)
        pdf.cell(col_widths[3], 7, f"{row.get('genuine_probability', 0):.2f}", border=1)
        pdf.cell(col_widths[4], 7, f"{row.get('fake_probability', 0):.2f}", border=1)
        pdf.ln()

    return bytes(pdf.output())


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ AI Review Guard")
    st.caption("v3.0.0 | Powered by ML + NLP")
    st.divider()

    api_status = _api_ok()
    if api_status:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Offline")
        st.info("Start backend with:\n```\npython api.py\n```")

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This system detects fake reviews using an ensemble ML model "
        "(LR + RF + XGBoost) trained with TF-IDF and behavioural signals, "
        "then extracts real customer insights from genuine feedback."
    )

# ── main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🔍 Single Review Check", "📦 Batch Analysis", "📊 Insights Dashboard"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Review Check
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Single Review Authenticator")
    st.markdown("Paste any product review to instantly detect if it's fake or genuine.")
    st.divider()

    review_text = st.text_area(
        "Review text",
        height=160,
        placeholder="e.g. 'This product exceeded all my expectations. Battery lasts all day and the build quality is excellent!'",
        key="single_review_input",
    )

    col_btn, col_sample = st.columns([1, 4])
    with col_btn:
        run_single = st.button("🔍 Analyse Review", use_container_width=True)
    with col_sample:
        if st.button("Load sample review"):
            st.session_state["single_review_input"] = (
                "Battery dies too fast and the screen flickers constantly. "
                "Very disappointed with the quality for the price paid."
            )
            st.rerun()

    if run_single:
        if not review_text or not review_text.strip():
            st.warning("⚠️ Please enter a review first.")
        else:
            with st.spinner("Analysing…"):
                data = _predict(review_text)

            if data:
                pred = data["prediction"]
                conf = data["confidence"]
                sent = data["sentiment"]
                bsig = data.get("behavioral_signals", {})

                st.divider()

                # ── Verdict badge ──────────────────────────────────────────
                if pred == "Genuine":
                    st.markdown(
                        f"""<div class="genuine-badge">
                        <h1 style="color:#3fb950;margin:0">✅ Genuine</h1>
                        <p style="color:#88d49b;font-size:1.1rem;margin:6px 0 0">
                        Confidence: <b>{conf:.1%}</b></p>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div class="fake-badge">
                        <h1 style="color:#f85149;margin:0">🚨 Fake</h1>
                        <p style="color:#ffa198;font-size:1.1rem;margin:6px 0 0">
                        Confidence: <b>{conf:.1%}</b></p>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.divider()
                col_a, col_b = st.columns(2)

                # ── Probability gauge ──────────────────────────────────────
                with col_a:
                    st.markdown("#### 🎯 Authenticity Probability")
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=data["genuine_probability"] * 100,
                            title={"text": "Genuine %", "font": {"color": "#c9d1d9"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                                "bar": {"color": "#2ea043"},
                                "bgcolor": "#161b22",
                                "steps": [
                                    {"range": [0, 40], "color": "#3d1b1b"},
                                    {"range": [40, 60], "color": "#3d3a1b"},
                                    {"range": [60, 100], "color": "#1b3d23"},
                                ],
                                "threshold": {
                                    "line": {"color": "#58a6ff", "width": 3},
                                    "thickness": 0.75,
                                    "value": 50,
                                },
                            },
                            number={"suffix": "%", "font": {"color": "#c9d1d9"}},
                        )
                    )
                    fig_gauge.update_layout(
                        paper_bgcolor="#0d1117",
                        height=260,
                        margin=dict(t=40, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # ── Sentiment radar ────────────────────────────────────────
                with col_b:
                    st.markdown("#### 🌡️ Sentiment Breakdown")
                    sent_fig = go.Figure(
                        go.Bar(
                            x=["Positive", "Neutral", "Negative"],
                            y=[sent["positive"], sent["neutral"], sent["negative"]],
                            marker_color=["#2ea043", "#8b949e", "#f85149"],
                            text=[f"{v:.2f}" for v in [sent["positive"], sent["neutral"], sent["negative"]]],
                            textposition="outside",
                        )
                    )
                    sent_fig.update_layout(
                        paper_bgcolor="#0d1117",
                        plot_bgcolor="#161b22",
                        font_color="#c9d1d9",
                        height=260,
                        margin=dict(t=20, b=20, l=20, r=20),
                        yaxis=dict(range=[0, 1.1], gridcolor="#21262d"),
                        xaxis=dict(gridcolor="#21262d"),
                    )
                    st.plotly_chart(sent_fig, use_container_width=True)
                    compound = sent["compound"]
                    label = sent["label"]
                    colour = {"Positive": "#2ea043", "Negative": "#f85149", "Neutral": "#8b949e"}[label]
                    st.markdown(
                        f"**Overall:** <span style='color:{colour};font-weight:700'>{label}</span>"
                        f" &nbsp;·&nbsp; Compound score: **{compound:+.4f}**",
                        unsafe_allow_html=True,
                    )

                # ── Behavioural signals ────────────────────────────────────
                st.divider()
                st.markdown("#### 🧪 Behavioural Signals")
                sig_cols = st.columns(len(bsig))
                labels_map = {
                    "review_length": "Length",
                    "word_count": "Words",
                    "exclamation_count": "Exclamations",
                    "caps_ratio": "Caps Ratio",
                    "avg_word_length": "Avg Word Len",
                    "punctuation_density": "Punct. Density",
                    "digit_ratio": "Digit Ratio",
                }
                for col, (k, v) in zip(sig_cols, bsig.items()):
                    col.metric(labels_map.get(k, k), f"{v:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📦 Batch Review Processor")
    st.markdown(
        "Upload a CSV file with a `review_text` column and get predictions for every row in the background."
    )
    st.divider()

    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], help="Must contain a `review_text` column."
    )

    if uploaded:
        raw_bytes = uploaded.read()
        df_preview = pd.read_csv(io.BytesIO(raw_bytes))
        st.info(f"📄 **{uploaded.name}** — {len(df_preview):,} rows loaded")
        st.dataframe(df_preview.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=False):
            result_rows = _batch_predict_csv(raw_bytes)

            if result_rows:
                df_result = pd.DataFrame(result_rows)
                st.divider()
                st.markdown("### 📋 Prediction Results")

                # ── Summary metrics ────────────────────────────────────────
                genuine_n = (df_result["prediction"] == "Genuine").sum()
                fake_n = (df_result["prediction"] == "Fake").sum()
                avg_conf = df_result["confidence"].mean()
                avg_compound = df_result["sentiment_compound"].mean()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Reviews", len(df_result))
                m2.metric("✅ Genuine", genuine_n)
                m3.metric("🚨 Fake", fake_n)
                m4.metric("Avg Confidence", f"{avg_conf:.1%}")

                # ── Authenticity donut ─────────────────────────────────────
                col_p, col_t = st.columns([1, 2])
                with col_p:
                    donut = px.pie(
                        values=[genuine_n, fake_n],
                        names=["Genuine", "Fake"],
                        hole=0.55,
                        color_discrete_sequence=["#2ea043", "#f85149"],
                        title="Authenticity Split",
                    )
                    donut.update_layout(
                        paper_bgcolor="#0d1117",
                        font_color="#c9d1d9",
                        height=300,
                        margin=dict(t=40, b=0, l=0, r=0),
                        legend=dict(bgcolor="#161b22"),
                    )
                    st.plotly_chart(donut, use_container_width=True)

                # ── Confidence distribution histogram ──────────────────────
                with col_t:
                    hist = px.histogram(
                        df_result,
                        x="confidence",
                        color="prediction",
                        nbins=20,
                        color_discrete_map={"Genuine": "#2ea043", "Fake": "#f85149"},
                        title="Confidence Distribution",
                        labels={"confidence": "Confidence Score", "count": "Count"},
                    )
                    hist.update_layout(
                        paper_bgcolor="#0d1117",
                        plot_bgcolor="#161b22",
                        font_color="#c9d1d9",
                        height=300,
                        legend=dict(bgcolor="#161b22"),
                        xaxis=dict(gridcolor="#21262d"),
                        yaxis=dict(gridcolor="#21262d"),
                    )
                    st.plotly_chart(hist, use_container_width=True)

                # ── Color-coded results table ──────────────────────────────
                st.markdown("#### Color-coded Predictions")

                def _row_color(row):
                    if row["prediction"] == "Fake":
                        return ["background-color: rgba(248,81,73,0.15)"] * len(row)
                    return ["background-color: rgba(46,160,67,0.10)"] * len(row)

                st.dataframe(
                    df_result.style.apply(_row_color, axis=1),
                    use_container_width=True,
                    height=400,
                )

                # ── Downloads ─────────────────────────────────────────────
                st.divider()
                dl1, dl2 = st.columns(2)
                with dl1:
                    csv_bytes = df_result.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv_bytes,
                        file_name="review_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with dl2:
                    try:
                        summary_pdf = {
                            "total": len(df_result),
                            "genuine": int(genuine_n),
                            "fake": int(fake_n),
                            "sentiment": "Positive" if avg_compound >= 0.05 else (
                                "Negative" if avg_compound <= -0.05 else "Neutral"
                            ),
                        }
                        pdf_bytes = _make_pdf(df_result, summary_pdf)
                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name="review_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except ImportError:
                        st.info("Install `fpdf2` to enable PDF export.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Insights Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Customer Insights Dashboard")
    st.markdown(
        "Paste a list of reviews (one per line) or use sample data to see "
        "complaint categorisation, sentiment distribution, and topic clusters."
    )
    st.divider()

    use_sample = st.checkbox("Use built-in sample reviews", value=True)
    if use_sample:
        raw_input = "\n".join(
            [
                "Battery dies too fast, can't last a full day.",
                "I wish there was a dark mode option in the app.",
                "The app keeps crashing when I open settings.",
                "Please add Google Drive integration, it's badly needed.",
                "Delivery took three weeks, extremely disappointed.",
                "Price is way too high for the quality you get.",
                "Screen flickers sometimes and the UI is confusing.",
                "Customer service was rude and completely unhelpful.",
                "Amazing product! Best purchase I've made this year.",
                "The build quality is solid and it looks premium.",
                "Fast shipping and excellent packaging, very impressed.",
                "Would love an export to CSV feature in future updates.",
                "Device gets very hot during long video calls.",
                "Sound quality from the speakers is crystal clear.",
                "Charging is extremely slow, needs fast charge support.",
            ]
        )
    else:
        raw_input = st.text_area(
            "Paste reviews (one per line)", height=200, placeholder="Review 1\nReview 2\n..."
        )

    if st.button("🔎 Generate Insights", use_container_width=False):
        reviews_list = [r.strip() for r in raw_input.strip().splitlines() if r.strip()]
        if not reviews_list:
            st.warning("⚠️ No reviews to analyse.")
        else:
            with st.spinner("Running analysis pipeline…"):
                data = _analyze(reviews_list)

            if data:
                genuine_n = data.get("genuine_count", 0)
                fake_n = data.get("fake_count", 0)
                total_n = data.get("total_submitted", len(reviews_list))
                sent_summary = data.get("sentiment_summary", {})
                clusters = data.get("topic_clusters", [])
                insights = data.get("insights", {})

                # ── Top-line metrics ───────────────────────────────────────
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Submitted", total_n)
                m2.metric("✅ Genuine", genuine_n)
                m3.metric("🚨 Fake Filtered", fake_n)
                avg_cpd = sent_summary.get("average_compound", 0)
                m4.metric(
                    "Avg Sentiment",
                    f"{avg_cpd:+.3f}",
                    delta=sent_summary.get("overall_label", ""),
                )

                col_left, col_right = st.columns(2)

                # ── Sentiment distribution ─────────────────────────────────
                with col_left:
                    st.markdown("#### 🌡️ Sentiment Distribution")
                    dist = sent_summary.get("distribution", {"Positive": 0, "Neutral": 0, "Negative": 0})
                    sent_bar = px.bar(
                        x=list(dist.keys()),
                        y=list(dist.values()),
                        color=list(dist.keys()),
                        color_discrete_map={
                            "Positive": "#2ea043",
                            "Neutral": "#8b949e",
                            "Negative": "#f85149",
                        },
                        labels={"x": "Sentiment", "y": "Review Count"},
                    )
                    sent_bar.update_layout(
                        paper_bgcolor="#0d1117",
                        plot_bgcolor="#161b22",
                        font_color="#c9d1d9",
                        showlegend=False,
                        height=280,
                        xaxis=dict(gridcolor="#21262d"),
                        yaxis=dict(gridcolor="#21262d"),
                        margin=dict(t=10, b=30, l=10, r=10),
                    )
                    st.plotly_chart(sent_bar, use_container_width=True)

                # ── Authenticity donut ─────────────────────────────────────
                with col_right:
                    st.markdown("#### 🥧 Authenticity Split")
                    donut2 = px.pie(
                        values=[genuine_n, fake_n],
                        names=["Genuine", "Fake"],
                        hole=0.6,
                        color_discrete_sequence=["#238636", "#da3633"],
                    )
                    donut2.update_layout(
                        paper_bgcolor="#0d1117",
                        font_color="#c9d1d9",
                        height=280,
                        margin=dict(t=10, b=0, l=0, r=0),
                        legend=dict(bgcolor="#161b22"),
                    )
                    st.plotly_chart(donut2, use_container_width=True)

                st.divider()

                # ── Complaint Breakdown ────────────────────────────────────
                st.markdown("#### 🔥 Top Complaint Categories")
                top_complaints = insights.get("top_complaints", [])
                if top_complaints:
                    complaint_df = pd.DataFrame(top_complaints)
                    if "category" in complaint_df.columns and "count" in complaint_df.columns:
                        bar = px.bar(
                            complaint_df,
                            x="count",
                            y="category",
                            orientation="h",
                            color="count",
                            color_continuous_scale=["#1e3a5f", "#f85149"],
                            labels={"count": "Number of complaints", "category": ""},
                        )
                        bar.update_layout(
                            paper_bgcolor="#0d1117",
                            plot_bgcolor="#161b22",
                            font_color="#c9d1d9",
                            height=320,
                            coloraxis_showscale=False,
                            margin=dict(t=10, b=10, l=10, r=10),
                            xaxis=dict(gridcolor="#21262d"),
                            yaxis=dict(gridcolor="#21262d"),
                        )
                        st.plotly_chart(bar, use_container_width=True)
                else:
                    st.info("No complaints categorised (all reviews may be positive).")

                # ── Feature Requests ───────────────────────────────────────
                feature_reqs = insights.get("feature_requests", [])
                if feature_reqs:
                    st.markdown("#### 💡 Feature Requests Detected")
                    for i, req in enumerate(feature_reqs, 1):
                        st.markdown(
                            f"<div style='background:rgba(88,166,255,0.08);border-left:3px solid #58a6ff;"
                            f"border-radius:6px;padding:10px 14px;margin:6px 0'>"
                            f"<b>{i}.</b> {req}</div>",
                            unsafe_allow_html=True,
                        )

                st.divider()

                # ── Topic Clusters ─────────────────────────────────────────
                st.markdown("#### 🗂️ Topic Clusters (Genuine Reviews Only)")
                if clusters:
                    for cluster in clusters:
                        with st.expander(
                            f"📌 Cluster {cluster['cluster_id']} — "
                            f"{cluster['size']} reviews | "
                            f"Top terms: {', '.join(cluster['key_terms'][:4])}"
                        ):
                            st.markdown(f"**All key terms:** `{' · '.join(cluster['key_terms'])}`")
                            st.markdown("**Example reviews:**")
                            for ex in cluster.get("representative_examples", []):
                                st.markdown(
                                    f"<div style='background:#161b22;border:1px solid #30363d;"
                                    f"border-radius:6px;padding:10px;margin:4px 0'>{ex}</div>",
                                    unsafe_allow_html=True,
                                )
                else:
                    st.info("Not enough genuine reviews for topic clustering.")

                # ── Summary metrics table ──────────────────────────────────
                st.divider()
                st.markdown("#### 📈 Summary Metrics")
                sm = insights.get("summary_metrics", {})
                smdf = pd.DataFrame(
                    [
                        {"Metric": "Total Reviews Analysed", "Value": sm.get("total_reviews", genuine_n)},
                        {"Metric": "Negative Review %", "Value": sm.get("negative_review_percentage", "N/A")},
                        {"Metric": "Average Sentiment Score", "Value": sm.get("average_sentiment", "N/A")},
                        {"Metric": "Overall Sentiment", "Value": sent_summary.get("overall_label", "N/A")},
                    ]
                )
                st.dataframe(smdf, use_container_width=True, hide_index=True)
