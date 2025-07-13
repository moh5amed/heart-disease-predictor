import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")
st.title("Heart Disease Risk Predictor")
st.write("Fill out the form below to assess your risk of heart disease.")

# Load model and preprocessing objects
model = joblib.load('best_heart_disease_model_P3.joblib')
scaler = joblib.load('scaler_P3.joblib')
tfidf = joblib.load('tfidf_vectorizer_P3.joblib')
categorical_cols = ['gender', 'smoking_status', 'physical_activity', 'chest_pain', 'shortness_breath', 'family_history']
label_encoders = {col: joblib.load(f'{col}_encoder_P3.joblib') for col in categorical_cols}

# User input form
def user_form():
    with st.form("risk_form"):
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["male", "female"])
        smoking_status = st.radio("Do you smoke?", ["yes", "no"])
        physical_activity = st.radio("Do you exercise regularly?", ["yes", "no"])
        chest_pain = st.radio("Do you have chest pain?", ["yes", "no"])
        shortness_breath = st.radio("Do you feel shortness of breath?", ["yes", "no"])
        family_history = st.radio("Family history of heart disease?", ["yes", "no"])
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        blood_pressure = st.number_input("Blood Pressure (Systolic, mmHg)", min_value=80.0, max_value=250.0, value=120.0)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100.0, max_value=500.0, value=200.0)
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50.0, max_value=400.0, value=90.0)
        max_heart_rate = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60.0, max_value=220.0, value=150.0)
        doctor_notes = st.text_area("Describe your symptoms or health history (optional)", "")
        submitted = st.form_submit_button("Predict Risk")
    return submitted, {
        'age': age,
        'gender': gender,
        'smoking_status': smoking_status,
        'physical_activity': physical_activity,
        'chest_pain': chest_pain,
        'shortness_breath': shortness_breath,
        'family_history': family_history,
        'weight': weight,
        'height': height,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'max_heart_rate': max_heart_rate,
        'doctor_notes': doctor_notes,
    }

def preprocess_input(user_data):
    # Encode categorical
    cat_features = [label_encoders[col].transform([user_data[col]])[0] for col in categorical_cols]
    # Numeric features
    numeric_cols = [
        'age', 'weight', 'height', 'blood_pressure', 'cholesterol',
        'fasting_blood_sugar', 'max_heart_rate'
    ]
    numeric_features = [float(user_data[col]) for col in numeric_cols]
    numeric_scaled = scaler.transform([numeric_features])[0]
    # TF-IDF for doctor_notes
    tfidf_features = tfidf.transform([user_data['doctor_notes']]).toarray()[0]
    # Combine all features in the correct order
    features = cat_features + list(numeric_scaled) + list(tfidf_features)
    return np.array(features).reshape(1, -1)

submitted, user_data = user_form()

if submitted:
    X = preprocess_input(user_data)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"Heart Disease Detected! Probability: {proba*100:.1f}%")
    else:
        st.success(f"No Heart Disease Detected. Probability: {proba*100:.1f}%")
    st.markdown("---")
    st.write("**Your Input:**")
    st.json(user_data) 