import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Review AI", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Product Review Guard")
st.markdown("Authenticate product reviews and extract actionable customer insights.")

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Sidebar for Batch Uploads
st.sidebar.header("Batch Analysis")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Text for Analytics", type=["csv", "txt"])

# Main Interface: Single Review Prediction
st.header("1. Individual Review Check")
review_text = st.text_area("Enter review text below:", height=150, placeholder="Example: This product is amazing, but the battery dies too quickly...")

if st.button("Detect Fake Review"):
    if review_text.strip():
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(f"{API_BASE_URL}/predict-review", json={"text": review_text})
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display Results
                    st.subheader("Result")
                    pred = data['prediction']
                    conf = data['probability_score']
                    
                    if pred == "Genuine":
                        st.success(f"**{pred}** (Confidence: {conf:.2%})")
                    else:
                        st.error(f"**{pred}** (Confidence: {conf:.2%})")
                    
                    # Sentiment Breakdown
                    st.write("**Sentiment Analysis:**")
                    sent = data['sentiment']
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Compound", f"{sent['compound']:.4f}")
                    col2.markdown(f"🟢 **Pos**: {sent['positive']:.2f}")
                    col3.markdown(f"🟡 **Neu**: {sent['neutral']:.2f}")
                    col4.markdown(f"🔴 **Neg**: {sent['negative']:.2f}")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
    else:
        st.warning("Please enter review text first.")

st.divider()

# Analytics Section
st.header("2. Insights & Analytics")

# Mock data/placeholder if no file is uploaded yet
if uploaded_file is None:
    st.info("💡 Upload a file in the sidebar to see batch analytics, or use the sample reviews below.")
    if st.checkbox("Show Sample Analytics"):
        sample_reviews = [
            "Battery dies too fast.", "Amazing quality, love it!", 
            "Scam product, do not buy.", "Total fraud, seller is fake.",
            "Screen flicker issues.", "Best purchase ever.",
            "Display is blurry.", "Charging takes forever.",
            "Fake review test.", "Five stars, perfect!"
        ]
        
        with st.spinner("Extracting insights..."):
            try:
                response = requests.post(f"{API_BASE_URL}/analyze-feedback", json={"reviews": sample_reviews})
                if response.status_code == 200:
                    data = response.json()
                    clusters = data['topic_clusters']
                    summary = data['sentiment_summary']
                    
                    # 1. Key Insights Summary
                    st.subheader("📌 Key Product Insights (Genuine Feedback Only)")
                    
                    readable = data.get('readable_insights', {})
                    if readable:
                        if 'message' in readable:
                            st.info(readable['message'])
                        else:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write("**Top Complaints:**")
                                for c in readable.get('top_complaints', []):
                                    st.warning(c)
                            with col_b:
                                st.write("**Most Requested Features:**")
                                for f in readable.get('feature_requests', []):
                                    st.success(f)
                    
                    st.write("**Common Feedback Themes (Clusters):**")
                    if clusters:
                        for cluster in clusters:
                            st.info(f"Theme {cluster['cluster_id']}: {', '.join(cluster['key_terms'])} ({cluster['size']} reviews)")
                    else:
                        st.write("No significant clusters detected in genuine reviews.")
                    
                    # 2. Visualizations
                    st.subheader("📊 Data Visualizations")
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.write("**Overall Sentiment Distribution**")
                        if 'genuine_count' in summary:
                            st.metric("Genuine Reviews Analyzed", f"{summary['genuine_count']} / {summary['total_count']}")
                        st.metric("Negative Review % (Genuine)", summary.get('negative_percentage', '0.0%'))
                        
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        dist = summary.get('distribution', {'Positive': 0, 'Neutral': 0, 'Negative': 0})
                        labels = list(dist.keys())
                        sizes = list(dist.values())
                        
                        if sum(sizes) > 0:
                            ax1.bar(labels, sizes, color=['#4CAF50', '#FFC107', '#F44336'])
                        else:
                            ax1.text(0.5, 0.5, 'No Data', ha='center', va='center')
                        st.pyplot(fig1)
                        
                    with c2:
                        st.write("**Authenticity Ratio (Fake vs Genuine)**")
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        if 'genuine_count' in summary:
                            g_cnt = summary['genuine_count']
                            f_cnt = summary['total_count'] - g_cnt
                            ax2.pie([g_cnt, f_cnt], labels=['Genuine', 'Fake'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
                        else:
                            ax2.pie([7, 3], labels=['Genuine', 'Fake'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
                        st.pyplot(fig2)
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
else:
    # Handle actual uploaded file logic here (Reading CSV/Text)
    try:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
            # Assuming 'review_text' column
            if 'review_text' in df_upload.columns:
                reviews = df_upload['review_text'].tolist()
                # Run the same analysis as sample above...
                st.success(f"File '{uploaded_file.name}' loaded successfully. Run analysis to see results.")
            else:
                st.error("CSV must contain a 'review_text' column.")
    except Exception as e:
        st.error(f"File Error: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("v2.0.0 | Powered by scikit-learn & VADER")
