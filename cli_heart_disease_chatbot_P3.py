import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Paths for preprocessing objects
categorical_cols = ['gender', 'smoking_status', 'physical_activity', 'chest_pain', 'shortness_breath', 'family_history']
le_paths = {col: f'{col}_encoder_P3.joblib' for col in categorical_cols}
scaler_path = 'scaler_P3.joblib'
tfidf_path = 'tfidf_vectorizer_P3.joblib'

# Load encoders, scaler, and vectorizer
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = joblib.load(le_paths[col])
scaler = joblib.load(scaler_path)
tfidf = joblib.load(tfidf_path)

# Load the trained model
model = joblib.load('best_heart_disease_model_P3.joblib')

# Feature prompts and types
feature_prompts = [
    ('age', 'Enter age (years): ', int),
    ('gender', "Enter gender ('male' or 'female'): ", str),
    ('smoking_status', "Do you smoke? ('yes' or 'no'): ", str),
    ('physical_activity', "Do you exercise regularly? ('yes' or 'no'): ", str),
    ('chest_pain', "Do you have chest pain? ('yes' or 'no'): ", str),
    ('shortness_breath', "Do you feel shortness of breath? ('yes' or 'no'): ", str),
    ('family_history', "Family history of heart disease? ('yes' or 'no'): ", str),
    ('weight', 'Enter weight (kg): ', float),
    ('height', 'Enter height (cm): ', float),
    ('blood_pressure', 'Enter blood pressure (systolic, mmHg): ', float),
    ('cholesterol', 'Enter cholesterol (mg/dL): ', float),
    ('fasting_blood_sugar', 'Enter fasting blood sugar (mg/dL): ', float),
    ('max_heart_rate', 'Enter max heart rate achieved (bpm): ', float),
    ('doctor_notes', 'Enter any additional notes (symptoms, history, etc.): ', str)
]

# Collect user input
def get_user_input():
    user_data = {}
    for key, prompt, typ in feature_prompts:
        while True:
            val = input(prompt)
            if val.strip() == '' or val.strip().lower() == 'unknown':
                # Handle missing/unknown values
                if typ in [int, float]:
                    val = 0  # Use 0 as a placeholder for missing numeric values
                else:
                    val = label_encoders[key].classes_[0] if key in label_encoders else ''
                print(f"No value entered for {key}, using default: {val}")
            try:
                if typ == int:
                    val = int(val)
                elif typ == float:
                    val = float(val)
                else:
                    val = str(val)
                user_data[key] = val
                break
            except ValueError:
                print(f"Invalid input for {key}. Please enter a valid {typ.__name__}.")
    return user_data

# Preprocess user input
def preprocess_input(user_data):
    # Encode categorical
    for col in categorical_cols:
        user_data[col] = label_encoders[col].transform([user_data[col]])[0]
    # Numeric features
    numeric_cols = [
        'age', 'weight', 'height', 'blood_pressure', 'cholesterol',
        'fasting_blood_sugar', 'max_heart_rate'
    ]
    numeric_vals = scaler.transform([[user_data[col] for col in numeric_cols]])[0]
    # TF-IDF for doctor_notes
    tfidf_vals = tfidf.transform([user_data['doctor_notes']]).toarray()[0]
    # Combine all features in the correct order
    features = [user_data[col] for col in categorical_cols] + list(numeric_vals) + list(tfidf_vals)
    return np.array(features).reshape(1, -1)

if __name__ == "__main__":
    print("\nWelcome to the Heart Disease Prediction Chatbot (P3)!\n")
    user_data = get_user_input()
    X_user = preprocess_input(user_data)
    pred = model.predict(X_user)[0]
    proba = model.predict_proba(X_user)[0][1]
    print("\nPrediction: {}".format('Heart Disease Detected' if pred == 1 else 'No Heart Disease Detected'))
    print(f"Probability of Heart Disease: {proba:.2f}") 