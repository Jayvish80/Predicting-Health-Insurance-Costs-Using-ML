import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Insurance Predictor", layout="centered")

st.title("Health Insurance Cost Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

data = load_data()

# Preprocessing
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

data["sex"] = le_sex.fit_transform(data["sex"])
data["smoker"] = le_smoker.fit_transform(data["smoker"])
data["region"] = le_region.fit_transform(data["region"])

X = data.drop("charges", axis=1)
y = data["charges"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    return model

model = train_model()

# ---------------- UI ----------------

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        children = st.number_input("Children", 0, 5, 0)

    with col2:
        sex = st.selectbox("Sex", le_sex.classes_)
        smoker = st.selectbox("Smoker", le_smoker.classes_)
        region = st.selectbox("Region", le_region.classes_)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [le_sex.transform([sex])[0]],
        "bmi": [bmi],
        "children": [children],
        "smoker": [le_smoker.transform([smoker])[0]],
        "region": [le_region.transform([region])[0]]
    })

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated Insurance Cost: ₹ {prediction:,.2f}")
