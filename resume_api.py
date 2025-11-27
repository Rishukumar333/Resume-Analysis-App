# ===========================
# resume_api.py (Final Improved Version ✅)
# ===========================

from flask import Flask, request, jsonify
import joblib
import os
import tempfile
import pdfplumber
import docx

app = Flask(__name__)

# -------------------------------------------------------------
# 🔄 Load models safely
# -------------------------------------------------------------
try:
    model_role = joblib.load("model_role.pkl")
    model_company = joblib.load("model_company.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder_role = joblib.load("label_encoder_role.pkl")
    label_encoder_company = joblib.load("label_encoder_company.pkl")
    print("✅ Models and encoders loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model_role = model_company = vectorizer = label_encoder_role = label_encoder_company = None

# -------------------------------------------------------------
# 🏠 Home route
# -------------------------------------------------------------
@app.route('/')
def home():
    return "✅ Resume Recommendation API is running!"

# -------------------------------------------------------------
# 📄 Text Extraction Functions
# -------------------------------------------------------------
def extract_text_from_pdf(path):
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages:
            t = pg.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

# -------------------------------------------------------------
# 🔍 Prediction Endpoint
# -------------------------------------------------------------
@app.route('/predict_file', methods=['POST'])
def predict_file():
    """
    Accepts multipart/form-data with a file field named 'file'.
    Supports: .pdf, .docx, .txt
    Returns JSON with predicted_job_role and recommended_company.
    """
    try:
        if not all([model_role, model_company, vectorizer]):
            return jsonify({"error": "Models not loaded on server"}), 500

        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        filename = file.filename or "uploaded_resume"

        # Save to temporary file
        suffix = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_name = tmp.name
            file.save(tmp_name)

        extracted_text = ""
        try:
            if suffix == ".pdf":
                extracted_text = extract_text_from_pdf(tmp_name)
            elif suffix in [".docx", ".doc"]:
                extracted_text = extract_text_from_docx(tmp_name)
            elif suffix == ".txt" or suffix == "":
                with open(tmp_name, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
            else:
                with open(tmp_name, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
        except Exception:
            return jsonify({"error": f"Unsupported or unreadable file: {suffix}"}), 400
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

        if not extracted_text.strip():
            return jsonify({"error": "Could not extract text from the uploaded file"}), 400

        # 🔠 Clean & preprocess text
        extracted_text = extracted_text.lower().strip()

        # Predict
        X_input = vectorizer.transform([extracted_text])
        role_pred = model_role.predict(X_input)[0]
        comp_pred = model_company.predict(X_input)[0]

        predicted_role = label_encoder_role.inverse_transform([role_pred])[0]
        recommended_company = label_encoder_company.inverse_transform([comp_pred])[0]

        return jsonify({
            "status": "success",
            "input_filename": filename,
            "predicted_job_role": predicted_role,
            "recommended_company": recommended_company
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------
# 🧪 Debug route
# -------------------------------------------------------------
@app.route('/debug')
def debug_info():
    return jsonify({
        "models_loaded": all([
            model_role is not None,
            model_company is not None,
            vectorizer is not None
        ]),
        "encoders_loaded": all([
            label_encoder_role is not None,
            label_encoder_company is not None
        ])
    })

# -------------------------------------------------------------
# 🚀 Run server
# -------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
