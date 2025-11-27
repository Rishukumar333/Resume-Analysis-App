# ===========================
# train_model.py (Final Optimized Version ✅)
# ===========================

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------------
# 🧠 Check if models already exist
# -------------------------------------------------------------
MODEL_FILES = [
    "model_role.pkl",
    "model_company.pkl",
    "tfidf_vectorizer.pkl",
    "label_encoder_role.pkl",
    "label_encoder_company.pkl"
]

if all(os.path.exists(f) for f in MODEL_FILES):
    print("✅ Models already exist — skipping training.")
    print("📂 Files found:")
    for f in MODEL_FILES:
        print("   ┗", f)
else:
    # -------------------------------------------------------------
    # 1️⃣ Load dataset
    # -------------------------------------------------------------
    data = pd.read_csv("resume_recommendation_dataset_filtered.csv")

    print("✅ Dataset loaded successfully!")
    print("📊 Shape of data:", data.shape)
    print("📁 Columns:", data.columns.tolist())

    # -------------------------------------------------------------
    # 2️⃣ Basic preprocessing
    # -------------------------------------------------------------
    # Keep only useful columns
    data = data[['skills', 'education', 'resume_summary', 
                 'predicted_job_role', 'recommended_companies', 
                 'industry_type', 'Recommended Skills']]

    # Drop missing rows for important columns
    data = data.dropna(subset=['skills', 'predicted_job_role', 'recommended_companies'])

    # 🏢 Handle multiple companies (keep first one)
    data['recommended_companies'] = data['recommended_companies'].apply(
        lambda x: x.split(',')[0].strip() if isinstance(x, str) else x
    )

    # Combine important resume features into one text field
    data['text_data'] = (
        data['skills'].astype(str) + " " +
        data['education'].astype(str) + " " +
        data['resume_summary'].astype(str) + " " +
        data['Recommended Skills'].astype(str) + " " +
        data['industry_type'].astype(str)
    )

    # -------------------------------------------------------------
    # 3️⃣ TF-IDF Vectorization
    # -------------------------------------------------------------
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['text_data'])

    # -------------------------------------------------------------
    # 4️⃣ Encode job roles
    # -------------------------------------------------------------
    label_encoder_role = LabelEncoder()
    y_role = label_encoder_role.fit_transform(data['predicted_job_role'])

    # -------------------------------------------------------------
    # 5️⃣ Encode companies
    # -------------------------------------------------------------
    label_encoder_company = LabelEncoder()
    y_company = label_encoder_company.fit_transform(data['recommended_companies'])

    # -------------------------------------------------------------
    # 6️⃣ Split data
    # -------------------------------------------------------------
    X_train, X_test, y_role_train, y_role_test, y_comp_train, y_comp_test = train_test_split(
        X, y_role, y_company, test_size=0.2, random_state=42
    )

    # -------------------------------------------------------------
    # 7️⃣ Train job role model
    # -------------------------------------------------------------
    model_role = RandomForestClassifier(n_estimators=100, random_state=42)
    model_role.fit(X_train, y_role_train)

    # -------------------------------------------------------------
    # 8️⃣ Train company model
    # -------------------------------------------------------------
    model_company = RandomForestClassifier(n_estimators=100, random_state=42)
    model_company.fit(X_train, y_comp_train)

    # -------------------------------------------------------------
    # 9️⃣ Evaluate both models
    # -------------------------------------------------------------
    y_role_pred = model_role.predict(X_test)
    y_comp_pred = model_company.predict(X_test)

    print("\n🎯 Job Role Prediction Accuracy:", accuracy_score(y_role_test, y_role_pred))
    print("\n📋 Job Role Classification Report:\n", classification_report(
        y_role_test, y_role_pred, target_names=label_encoder_role.classes_
    ))

    print("\n🏢 Company Recommendation Accuracy:", accuracy_score(y_comp_test, y_comp_pred))

    # Handle possible label mismatch safely
    try:
        print("\n📋 Company Recommendation Report:\n", classification_report(
            y_comp_test, y_comp_pred,
            labels=range(len(label_encoder_company.classes_)),
            target_names=label_encoder_company.classes_,
            zero_division=0
        ))
    except ValueError as e:
        print("\n⚠️ Skipping detailed company report due to label mismatch.")
        print("Error:", e)

    # -------------------------------------------------------------
    # 🔒 Save all models and encoders
    # -------------------------------------------------------------
    joblib.dump(model_role, "model_role.pkl")
    joblib.dump(model_company, "model_company.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder_role, "label_encoder_role.pkl")
    joblib.dump(label_encoder_company, "label_encoder_company.pkl")

    print("\n✅✅ Training completed and all models saved successfully! 🎉")

# -------------------------------------------------------------
# 🧩 Optional: Test loading models
# -------------------------------------------------------------
try:
    model_role = joblib.load("model_role.pkl")
    model_company = joblib.load("model_company.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder_role = joblib.load("label_encoder_role.pkl")
    label_encoder_company = joblib.load("label_encoder_company.pkl")
    print("\n🚀 Models loaded and ready to use for predictions!")
except Exception as e:
    print("❌ Error while loading models:", e)
