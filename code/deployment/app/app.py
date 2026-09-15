"""Stage 3: Web application — collects input and displays predictions from the API."""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")

single_tab, batch_tab = st.tabs(["Single passenger", "Upload CSV"])

with single_tab:
    with st.form("single_prediction"):
        pclass = st.selectbox("Passenger class", options=[1, 2, 3], index=2)
        sex = st.selectbox("Sex", options=["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        sibsp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input("Parents/children aboard", min_value=0, max_value=10, value=0, step=1)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        embarked = st.selectbox("Port of embarkation", options=["S", "C", "Q"])
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
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
                st.success(f"Survived — probability {result['survival_probability']:.0%}")
            else:
                st.error(f"Did not survive — probability of survival {result['survival_probability']:.0%}")
        except requests.exceptions.RequestException as exc:
            st.warning(f"Could not reach the model API: {exc}")

with batch_tab:
    st.caption("CSV must contain columns: " + ", ".join(FEATURE_COLUMNS))
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        missing_columns = [c for c in FEATURE_COLUMNS if c not in data.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        elif st.button("Run predictions", type="primary"):
            try:
                records = data[FEATURE_COLUMNS].to_dict(orient="records")
                response = requests.post(f"{API_URL}/predict_batch", json=records, timeout=60)
                response.raise_for_status()
                results = response.json()

                data["survived"] = [r["survived"] for r in results]
                data["survival_probability"] = [r["survival_probability"] for r in results]

                st.dataframe(data)
                st.download_button(
                    "Download predictions as CSV",
                    data.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except requests.exceptions.RequestException as exc:
                st.warning(f"Could not reach the model API: {exc}")
