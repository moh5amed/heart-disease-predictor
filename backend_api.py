from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and preprocessing objects
model = joblib.load('best_heart_disease_model_P3.joblib')
scaler = joblib.load('scaler_P3.joblib')
tfidf = joblib.load('tfidf_vectorizer_P3.joblib')
categorical_cols = ['gender', 'smoking_status', 'physical_activity', 'chest_pain', 'shortness_breath', 'family_history']
label_encoders = {col: joblib.load(f'{col}_encoder_P3.joblib') for col in categorical_cols}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract and preprocess features
    try:
        # Categorical
        cat_features = [label_encoders[col].transform([data[col]])[0] for col in categorical_cols]
        # Numeric
        numeric_cols = [
            'age', 'weight', 'height', 'blood_pressure', 'cholesterol',
            'fasting_blood_sugar', 'max_heart_rate'
        ]
        numeric_features = [float(data.get(col, 0)) for col in numeric_cols]
        numeric_scaled = scaler.transform([numeric_features])[0]
        # Text (doctor_notes/symptoms)
        doctor_notes = data.get('doctor_notes', '')
        tfidf_features = tfidf.transform([doctor_notes]).toarray()[0]
        # Combine all features in the correct order
        features = cat_features + list(numeric_scaled) + list(tfidf_features)
        X = np.array(features).reshape(1, -1)
        # Predict
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        return jsonify({
            'prediction': pred,
            'probability': proba
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 