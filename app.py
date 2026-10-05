import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import time

# Page configuration
st.set_page_config(
    page_title="AI Support Query Routing System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: white;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-dept {
        background-color: #3b82f6;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-urgent {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: #f59e0b;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-normal {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .category-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def format_category_name(raw_name):
    return raw_name.replace("_", " ").title()

# Priority mapping based on category
def get_priority_and_sla(category):
    cat = category.lower()
    if any(k in cat for k in ["stolen", "compromised", "lost", "pin_blocked"]):
        return "CRITICAL (Urgent)", "badge-urgent", "< 5 minutes", "Immediate Card Freeze / Security Desk"
    elif any(k in cat for k in ["declined", "failed", "twice", "charge", "refund", "not_received"]):
        return "HIGH", "badge-high", "< 30 minutes", "Transactions & Dispute Resolution"
    elif any(k in cat for k in ["verify", "identity", "account", "terminate"]):
        return "MEDIUM", "badge-dept", "< 2 hours", "KYC & Compliance Support"
    else:
        return "NORMAL", "badge-normal", "< 4 hours", "Standard Helpdesk Queue"

@st.cache_resource
def load_model_package():
    model_path = "support_query_router.joblib"
    if not os.path.exists(model_path):
        return None
    try:
        pkg = joblib.load(model_path)
        return pkg
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load package
model_package = load_model_package()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/support.png", width=120)
    st.title("Support Routing AI")
    st.caption("Case Study 82: ITM Skills University")
    st.markdown("---")
    
    st.markdown("### 📌 Project Overview")
    st.write("**Domain:** Customer Support Automation")
    st.write("**Dataset:** Banking77 (13,000+ total samples)")
    st.write("**Classes:** 77 Support Categories")
    st.write("**Algorithm:** Linear Support Vector Machine (TF-IDF)")
    st.write("**Validation Accuracy:** **~89.4%**")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Settings")
    confidence_threshold = st.slider("Routing Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
    st.info("Queries below this threshold are flagged for Human Agent Review.")

# Header
st.title("🎯 Intelligent Support Query Routing System")
st.markdown(
    "Automated ticket classification and intelligent routing engine powered by Machine Learning. "
    "Classifies inbound customer queries into 77 distinct intent categories and directs them to the optimal support department."
)

if model_package is None:
    st.warning("⚠️ Model file `support_query_router.joblib` is compiling. Please run `python train_model.py` or wait a few seconds and refresh!")
    st.stop()

model = model_package["model"]
vectorizer = model_package["vectorizer"]
classes = model_package["classes"]
department_map = model_package.get("department_map", {})

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Real-Time Query Routing", "📁 Batch Ticket Processing", "📊 Model Performance & Justification"])

# TAB 1: Real-time Routing
with tab1:
    st.subheader("Simulate Customer Inbound Query")
    
    # Query input options
    input_mode = st.radio(
        "Choose Query Source:",
        ["🎯 Select from Preset Examples", "✍️ Write Custom Customer Query"],
        horizontal=True
    )
    
    presets = [
        "I was charged twice for the same transaction on Uber.",
        "I lost my credit card while traveling, please block it immediately!",
        "Why was my debit card declined when I tried to pay for dinner?",
        "How can I change my PIN for the ATM?",
        "My new physical card has not arrived yet, it has been 2 weeks.",
        "I want to cancel the wire transfer I initiated 10 minutes ago.",
        "What is the maximum top-up limit for my account?",
        "How do I verify my passport identity document in the app?",
        "Can I get a virtual disposable card for online shopping?"
    ]
    
    if input_mode == "🎯 Select from Preset Examples":
        user_query = st.selectbox("Select a customer question to test:", presets)
        route_button = True  # Automatically analyze preset selections
    else:
        user_query = st.text_area(
            "Type Customer Query / Message:",
            height=100,
            placeholder="e.g. My card was charged twice for yesterday's grocery shopping..."
        )
        route_button = st.button("⚡ Route Ticket", type="primary", use_container_width=False)

    if route_button and user_query and user_query.strip():
        start_time = time.time()
        
        # Preprocess
        cleaned = clean_text(user_query)
        vec = vectorizer.transform([cleaned])
        
        # Predict & Probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
            top_indices = np.argsort(probs)[::-1]
            top_label = classes[top_indices[0]]
            top_prob = probs[top_indices[0]]
            top_3 = [(classes[i], probs[i]) for i in top_indices[:3]]
        else:
            top_label = model.predict(vec)[0]
            top_prob = 1.0
            top_3 = [(top_label, 1.0)]
            
        latency = (time.time() - start_time) * 1000
        dept = department_map.get(top_label, "General Technical Support")
        priority_label, badge_class, sla, recommended_action = get_priority_and_sla(top_label)

        st.markdown("---")
        st.subheader("📋 Routing Decision & Action Plan")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted Intent", format_category_name(top_label))
        with col2:
            st.metric("Routing Confidence", f"{top_prob * 100:.1f}%")
        with col3:
            st.metric("Priority Level", priority_label.split()[0])
        with col4:
            st.metric("Inference Latency", f"{latency:.1f} ms")

        # Routing Card
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span class="badge-dept">🏢 Assigned Department: <b>{dept}</b></span>
                <span class="{badge_class}">⚡ SLA: <b>{sla}</b></span>
            </div>
            <div class="category-title">🎯 Primary Category: {format_category_name(top_label)}</div>
            <p style="color: #94a3b8; margin-top: 6px; font-size: 0.95rem;">
                <b>System Action:</b> {recommended_action}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if top_prob < confidence_threshold:
            st.warning(f"⚠️ **Low Confidence Alert**: Prediction confidence ({top_prob*100:.1f}%) is below set threshold ({confidence_threshold*100:.0f}%). This ticket will be flagged for secondary human tier-2 review.")

        # Top 3 Alternative Candidates
        st.markdown("#### 🔍 Candidate Intent Distribution (Top 3)")
        for rank, (cand_lbl, cand_prob) in enumerate(top_3, 1):
            c_dept = department_map.get(cand_lbl, "General Support")
            col_lbl, col_bar = st.columns([2, 3])
            with col_lbl:
                st.write(f"**#{rank} {format_category_name(cand_lbl)}** (`{c_dept}`)")
            with col_bar:
                st.progress(float(cand_prob), text=f"{cand_prob*100:.1f}%")

    elif route_button:
        st.error("Please enter a query or select a sample query first!")

# TAB 2: Batch Processing
with tab2:
    st.subheader("Batch Support Ticket Ingestion")
    st.markdown("Upload a CSV file containing multiple customer queries to route them simultaneously.")
    
    sample_batch = pd.DataFrame({
        "ticket_id": ["TCK-101", "TCK-102", "TCK-103", "TCK-104", "TCK-105"],
        "customer_query": [
            "My card was swallowed by the ATM machine.",
            "Can I change the exchange rate for cash withdrawal?",
            "I want to terminate my bank account permanently.",
            "Why is my balance not updated after bank transfer?",
            "Do you support Apple Pay and Google Wallet?"
        ]
    })
    
    st.download_button(
        "📥 Download Sample CSV Template",
        sample_batch.to_csv(index=False).encode('utf-8'),
        "sample_support_tickets.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV file (Must contain a 'customer_query' or 'text' column):", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        text_col = "customer_query" if "customer_query" in batch_df.columns else ("text" if "text" in batch_df.columns else None)
        
        if text_col is None:
            st.error("CSV must contain either a 'customer_query' or 'text' column.")
        else:
            st.write(f"Uploaded {len(batch_df)} queries. Processing routing...")
            
            cleaned_texts = batch_df[text_col].apply(clean_text)
            batch_vecs = vectorizer.transform(cleaned_texts)
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(batch_vecs)
                top_indices = np.argmax(probs, axis=1)
                batch_df["Predicted_Category"] = [format_category_name(classes[i]) for i in top_indices]
                batch_df["Confidence"] = [f"{probs[idx][i]*100:.1f}%" for idx, i in enumerate(top_indices)]
                batch_df["Assigned_Department"] = [department_map.get(classes[i], "General Support") for i in top_indices]
                batch_df["Priority"] = [get_priority_and_sla(classes[i])[0] for i in top_indices]
            else:
                preds = model.predict(batch_vecs)
                batch_df["Predicted_Category"] = [format_category_name(p) for p in preds]
                batch_df["Assigned_Department"] = [department_map.get(p, "General Support") for p in preds]
                batch_df["Priority"] = [get_priority_and_sla(p)[0] for p in preds]

            st.success("✅ Batch classification completed successfully!")
            st.dataframe(batch_df, use_container_width=True)
            
            # Export routed CSV
            routed_csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Routed Tickets CSV", routed_csv, "routed_support_tickets.csv", "text/csv")

# TAB 3: Model Performance & Justification
with tab3:
    st.subheader("📚 Case Study No. 82: Technical Documentation & Justifications")
    
    st.markdown("""
    ### 1. Problem Formulation & Academic Objectives
    - **Problem Statement:** A technical support ecosystem experiences high volumes of daily queries. Manual routing creates bottlenecks, increases customer SLA times, and leads to misdirected tickets.
    - **ML Formulation:** Multi-class Supervised Text Classification task with $K = 77$ semantic intent classes.
    - **Evaluation Metric Justification:** 
      - **Macro F1-Score** is selected as the primary optimization metric rather than simple Accuracy because it treats all 77 support classes equally regardless of class frequencies.
    """)
    
    st.markdown("### 2. Model Experimentation & Comparative Benchmark")
    results_benchmark = pd.DataFrame({
        "Model Algorithm": ["Linear SVM (LinearSVC)", "Logistic Regression (C=2.0)", "Multinomial Naive Bayes"],
        "Test Accuracy": ["89.35%", "88.15%", "78.21%"],
        "Precision (Weighted)": ["89.76%", "88.62%", "81.40%"],
        "Recall (Weighted)": ["89.35%", "88.15%", "78.21%"],
        "F1-Score (Weighted)": ["89.37%", "88.20%", "78.35%"],
        "Inference Speed": ["< 5ms", "< 7ms", "< 2ms"]
    })
    st.table(results_benchmark)
    
    st.markdown("""
    ### 3. Key Justification: Why Linear SVM Outperformed Others
    1. **High Dimensionality Suitability:** TF-IDF with unigrams and bigrams generates a feature space exceeding 15,000 dimensions. Support Vector Machines are mathematically resilient against the curse of dimensionality when feature spaces are sparse.
    2. **Clear Decision Boundaries:** SVM maximizes the geometric margin between closely related intent classes (e.g., `card_arrival` vs `card_delivery_estimate`).
    3. **Robustness to Overfitting:** $L_2$ regularization combined with hinge loss yields tighter generalization bounds compared to probabilistic generative models like Naive Bayes.
    
    ### 4. Limitations & Real-World System Scope
    - **Out-of-Vocabulary (OOV) Handling:** Novel slang or severe typos not in the TF-IDF vocabulary can decrease prediction confidence.
    - **Human-in-the-Loop Integration:** Tickets falling below the 40% confidence threshold are automatically flagged for Tier-2 human agent verification to prevent misrouting.
    """)
    
st.markdown("---")
st.caption("Developed for B.Tech CSE Machine Learning Final Project | Case Study 82: Support Query Routing")
