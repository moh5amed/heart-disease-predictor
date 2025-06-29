import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

# Load the dataset
df = pd.read_csv('heart_disease_data_P3.csv')

# Handle missing values (if any)
df = df.ffill().bfill()

# Encode categorical variables
categorical_cols = ['gender', 'smoking_status', 'physical_activity', 'chest_pain', 'shortness_breath', 'family_history']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    joblib.dump(le, f'{col}_encoder_P3.joblib')

# Scale numeric features
numeric_cols = [
    'age', 'weight', 'height', 'blood_pressure', 'cholesterol',
    'fasting_blood_sugar', 'max_heart_rate'
]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
joblib.dump(scaler, 'scaler_P3.joblib')

# Vectorize the doctor_notes text feature
tfidf = TfidfVectorizer(max_features=100)
doctor_notes_tfidf = tfidf.fit_transform(df['doctor_notes']).toarray()
tfidf_feature_names = [f'tfidf_{i}' for i in range(doctor_notes_tfidf.shape[1])]
doctor_notes_df = pd.DataFrame(doctor_notes_tfidf, columns=tfidf_feature_names)
joblib.dump(tfidf, 'tfidf_vectorizer_P3.joblib')

# Combine all features (drop original doctor_notes)
df_processed = pd.concat([df.drop(['doctor_notes'], axis=1).reset_index(drop=True), doctor_notes_df], axis=1)

# Separate features and target
y = df_processed['heart_disease']
X = df_processed.drop('heart_disease', axis=1)

# Split into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits as CSV files
X_train.to_csv('X_train_P3.csv', index=False)
X_val.to_csv('X_val_P3.csv', index=False)
y_train.to_csv('y_train_P3.csv', index=False)
y_val.to_csv('y_val_P3.csv', index=False)

print('Preprocessing and split complete. Files saved: X_train_P3.csv, X_val_P3.csv, y_train_P3.csv, y_val_P3.csv') 