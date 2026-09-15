"""Stage 3: Web application — collects input and displays predictions from the API."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details and click **Predict** to see the model's estimate.")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger class", options=[1, 2, 3], index=2)
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", options=["S", "C", "Q"])

with col2:
    sibsp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents/children aboard", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

if st.button("Predict", type="primary"):
    payload = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result["survived"]:
            st.success(f"Prediction: **Survived** (probability: {result['survival_probability']:.2%})")
        else:
            st.error(f"Prediction: **Did not survive** (probability of survival: {result['survival_probability']:.2%})")
    except requests.exceptions.RequestException as exc:
        st.warning(f"Could not reach the model API at {API_URL}: {exc}")
