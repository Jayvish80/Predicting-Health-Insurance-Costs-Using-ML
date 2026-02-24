import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Insurance Predictor", layout="centered")

st.title("Health Insurance Payment Prediction")

# Check files exist
required_files = [
    "best_model.pkl",
    "scaler.pkl",
    "label_encoder_gender.pkl",
    "label_encoder_diabetic.pkl",
    "label_encoder_smoker.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"Missing file: {file}")
        st.stop()

# Load models
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        children = st.number_input("Children", 0, 8, 0)

    with col2:
        bloodpressure = st.number_input("Blood Pressure", 60, 200, 120)
        gender = st.selectbox("Gender", le_gender.classes_)
        diabetic = st.selectbox("Diabetic", le_diabetic.classes_)
        smoker = st.selectbox("Smoker", le_smoker.classes_)

    submit = st.form_submit_button("Predict")

if submit:
    data = pd.DataFrame({
        "age": [age],
        "gender": le_gender.transform([gender]),
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": le_diabetic.transform([diabetic]),
        "children": [children],
        "smoker": le_smoker.transform([smoker])
    })

    num_cols = ["age", "bmi", "bloodpressure", "children"]
    data[num_cols] = scaler.transform(data[num_cols])

    prediction = model.predict(data)[0]

    st.success(f"Estimated Insurance Cost: ₹ {prediction:,.2f}")
