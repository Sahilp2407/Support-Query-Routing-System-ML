# 🤖 Intelligent Support Query Routing System (ML)

An end-to-end Machine Learning system that automatically classifies and routes incoming customer banking support queries into **77 intent categories** with **89.35% accuracy** and sub-5ms inference latency.

---

## 📌 Project Overview
- **Domain:** Customer Support Automation / Banking NLP
- **Dataset:** Banking77 (13,000+ total samples across 77 categories)
- **Primary Algorithm:** Linear Support Vector Machine (`LinearSVC`) + TF-IDF Vectorizer
- **Primary Metric:** Macro F1-Score & Accuracy (~89.35%)
- **Deployment:** Interactive Streamlit Dashboard with Real-Time Routing, Batch Processing, and Confidence Fallback Mechanisms.

---

## 🏗️ Architecture & Pipeline
1. **Data Ingestion & Cleaning:** Regex-based URL removal, punctuation stripping, case folding, and whitespace normalization.
2. **Feature Extraction:** Sublinear TF-IDF vectorization with unigram and bigram features (`ngram_range=(1, 2)`).
3. **Model Benchmarking:** Comparative analysis between:
   - **Linear SVM (LinearSVC)** — *Selected Best Model (89.35% Test Accuracy, 89.37% F1)*
   - **Logistic Regression (C=2.0)** — *88.15% Test Accuracy*
   - **Multinomial Naive Bayes (alpha=0.1)** — *78.21% Test Accuracy*
4. **Error Analysis & Retraining:** Evaluated semantic edge cases on validation data and retrained on 100% training samples for maximum vocabulary coverage.
5. **Model Serialization:** Packaged trained model and fitted vectorizer together into `support_query_router.joblib`.
6. **Production Deployment:** Streamlit application providing real-time single query inference, bulk CSV batch processing, and human-in-the-loop fallback routing (< 40% confidence).

---

## 📊 Performance Benchmark

| Model | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) | Inference Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **🥇 Linear SVM (LinearSVC)** | **89.35%** | **89.76%** | **89.35%** | **89.37%** | **< 5ms** |
| 🥈 Logistic Regression | 88.15% | 88.62% | 88.15% | 88.20% | < 7ms |
| 🥉 Multinomial Naive Bayes | 78.21% | 81.40% | 78.21% | 78.35% | < 2ms |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Sahilp2407/Support-Query-Routing-System-ML.git
cd Support-Query-Routing-System-ML
```

### 2. Set up virtual environment & dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or install core packages:
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure
```
├── app.py                         # Streamlit Web Application
├── Support_Query_Routing.ipynb   # Complete ML Training & Analysis Notebook
├── train_model.py                 # Standalone script to train & export model
├── support_query_router.joblib    # Serialized model + TF-IDF package
├── train.csv                      # Training dataset (10,003 rows)
├── test.csv                       # Test dataset (3,080 rows)
├── dev.csv                        # Validation / Dev dataset (3,080 rows)
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## 👤 Author
**Sahil Pandey**  
Roll No: 150096724024 | Cohort: Elon Musk  
B.Tech Computer Science & Engineering
