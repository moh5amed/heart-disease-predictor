import random
import pandas as pd
import time
import google.generativeai as genai

# Gemini API setup
API_KEY = "AIzaSyANR8M9lt7O8HIPmIIPFSUoC4VDqlF1GBE" # Replace with your actual key if needed
genai.configure(api_key=API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Features to generate
FEATURES = [
    'age', 'gender', 'smoking_status', 'physical_activity', 'chest_pain', 'shortness_breath',
    'family_history', 'weight', 'height', 'blood_pressure', 'cholesterol',
    'fasting_blood_sugar', 'max_heart_rate', 'doctor_notes', 'heart_disease'
]

# Helper functions for random feature generation
def random_gender():
    return random.choice(['male', 'female'])

def random_smoking():
    return random.choice(['yes', 'no'])

def random_physical_activity():
    return random.choice(['yes', 'no'])

def random_chest_pain():
    return random.choice(['yes', 'no'])

def random_shortness_breath():
    return random.choice(['yes', 'no'])

def random_family_history():
    return random.choice(['yes', 'no'])

def random_weight():
    return random.randint(50, 120)  # kg

def random_height():
    return random.randint(150, 200)  # cm

def random_blood_pressure():
    return random.randint(90, 180)  # systolic

def random_cholesterol():
    return random.randint(120, 350)  # mg/dL

def random_fasting_blood_sugar():
    return random.randint(70, 200)  # mg/dL

def random_max_heart_rate():
    return random.randint(70, 202)  # bpm

# Generate a single record
def generate_record(heart_disease):
    # For healthy vs. diseased, bias the features accordingly
    if heart_disease == 1:
        # Diseased: more risk factors
        age = random.randint(50, 80)
        gender = random_gender()
        smoking_status = random.choices(['yes', 'no'], weights=[0.7, 0.3])[0]
        physical_activity = random.choices(['no', 'yes'], weights=[0.7, 0.3])[0]
        chest_pain = random.choices(['yes', 'no'], weights=[0.8, 0.2])[0]
        shortness_breath = random.choices(['yes', 'no'], weights=[0.7, 0.3])[0]
        family_history = random.choices(['yes', 'no'], weights=[0.6, 0.4])[0]
        weight = random.randint(70, 120)
        height = random.randint(150, 180)
        blood_pressure = random.randint(140, 180)
        cholesterol = random.randint(220, 350)
        fasting_blood_sugar = random.randint(110, 200)
        max_heart_rate = random.randint(70, 140)
    else:
        # Healthy: fewer risk factors
        age = random.randint(20, 60)
        gender = random_gender()
        smoking_status = random.choices(['no', 'yes'], weights=[0.8, 0.2])[0]
        physical_activity = random.choices(['yes', 'no'], weights=[0.8, 0.2])[0]
        chest_pain = random.choices(['no', 'yes'], weights=[0.9, 0.1])[0]
        shortness_breath = random.choices(['no', 'yes'], weights=[0.9, 0.1])[0]
        family_history = random.choices(['no', 'yes'], weights=[0.7, 0.3])[0]
        weight = random.randint(50, 90)
        height = random.randint(160, 200)
        blood_pressure = random.randint(90, 130)
        cholesterol = random.randint(120, 210)
        fasting_blood_sugar = random.randint(70, 110)
        max_heart_rate = random.randint(120, 202)

    # Generate a user-friendly doctor note
    prompt = (
        f"Write a user-friendly doctor's note for a patient with the following information: "
        f"age {age}, gender {gender}, smoking status {smoking_status}, physical activity {physical_activity}, "
        f"chest pain {chest_pain}, shortness of breath {shortness_breath}, family history {family_history}, "
        f"weight {weight} kg, height {height} cm, blood pressure {blood_pressure} mmHg, cholesterol {cholesterol} mg/dL, "
        f"fasting blood sugar {fasting_blood_sugar} mg/dL, max heart rate {max_heart_rate} bpm. "
        f"Summarize the patient's health and risk of heart disease in simple terms."
    )
    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            doctor_notes = response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            doctor_notes = response.candidates[0].text.strip()
        else:
            doctor_notes = "No note generated."
    except Exception as e:
        print(f"Error generating note: {e}")
        doctor_notes = "Error generating note."

    return [
        age, gender, smoking_status, physical_activity, chest_pain, shortness_breath,
        family_history, weight, height, blood_pressure, cholesterol,
        fasting_blood_sugar, max_heart_rate, doctor_notes, heart_disease
    ]

def main(num_samples=40, output_csv='heart_disease_data_P3.csv'):
    data = []
    # Ensure balanced classes
    num_diseased = num_samples // 2
    num_healthy = num_samples - num_diseased
    print("Generating diseased cases...")
    for _ in range(num_diseased):
        data.append(generate_record(1))
        time.sleep(1.2)
    print("Generating healthy cases...")
    for _ in range(num_healthy):
        data.append(generate_record(0))
        time.sleep(1.2)
    df = pd.DataFrame(data, columns=FEATURES)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

if __name__ == "__main__":
    main(num_samples=40)  # Reduced to 40 to fit within Gemini API free tier quota 