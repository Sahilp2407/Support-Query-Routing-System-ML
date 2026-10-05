import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_and_export():
    print("Loading datasets...")
    train_df = pd.read_csv("train.csv")
    
    print(f"Cleaning training data ({len(train_df)} samples)...")
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    
    X = train_df["clean_text"]
    y = train_df["label_text"]
    
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    X_tfidf = tfidf.fit_transform(X)
    
    print("Training Linear SVM with probability calibration...")
    # CalibratedClassifierCV wraps LinearSVC with Platt scaling so predict_proba works seamlessly for confidence scores in Streamlit!
    base_svm = LinearSVC(C=1.0, random_state=42)
    calibrated_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_model.fit(X_tfidf, y)
    
    classes = list(calibrated_model.classes_)
    
    # Category human-friendly mapping (e.g. 'card_arrival' -> 'Card Arrival & Delivery')
    # and Department routing mapping (e.g. Cards, Transactions, Security, Account, Charges/Fees)
    def map_department(label):
        lbl = label.lower()
        if any(k in lbl for k in ["card", "pin", "contactless", "atm"]):
            return "Card Operations & Hardware"
        elif any(k in lbl for k in ["transfer", "payment", "beneficiary", "direct_debit", "apple_pay", "google_pay", "cash_withdrawal"]):
            return "Payments & Fund Transfers"
        elif any(k in lbl for k in ["charge", "fee", "refund", "exchange", "rate", "currency", "statement", "twice"]):
            return "Billing, Refunds & Currency Exchange"
        elif any(k in lbl for k in ["verify", "identity", "passcode", "compromised", "stolen", "lost"]):
            return "Security, KYC & Account Protection"
        elif any(k in lbl for k in ["top_up", "deposit", "cheque", "balance"]):
            return "Top-Up & Deposits"
        elif any(k in lbl for k in ["account", "terminate", "edit_personal_details", "age_limit", "country_support"]):
            return "Account Management & Support"
        else:
            return "General Customer Support"

    department_map = {c: map_department(c) for c in classes}

    model_package = {
        "model": calibrated_model,
        "vectorizer": tfidf,
        "model_name": "Linear Support Vector Machine (LinearSVC)",
        "classes": classes,
        "department_map": department_map
    }
    
    output_path = "support_query_router.joblib"
    print(f"Saving package to {output_path}...")
    joblib.dump(model_package, output_path)
    print("Done! Model successfully trained and saved.")

if __name__ == "__main__":
    train_and_export()
