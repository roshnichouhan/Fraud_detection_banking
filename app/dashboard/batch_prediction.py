import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder


# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("models/best_model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    return model, scaler


# ==========================================================
# Batch Prediction Page
# ==========================================================

def show_batch_prediction():

    st.title("📁 Batch Fraud Prediction")

    st.markdown(
        """
Upload a CSV file containing banking transactions.

The application will automatically:

- Validate the dataset
- Preprocess the data
- Encode categorical features
- Scale numerical features
- Predict fraud
- Display fraud probability
- Allow downloading predictions
"""
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(

        "Upload CSV",

        type=["csv"]

    )

    if uploaded_file is None:
        return

    # ======================================================
    # Load CSV
    # ======================================================

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # ======================================================
    # Check Required Columns
    # ======================================================

    if "label" in df.columns:

        df = df.drop("label", axis=1)

    # ======================================================
    # Encode Object Columns
    # ======================================================

    categorical_columns = df.select_dtypes(

        include="object"

    ).columns

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(

            df[column].astype(str)

        )

    # ======================================================
    # Load Model
    # ======================================================

    model, scaler = load_artifacts()

    # ======================================================
    # Scale Features
    # ======================================================

    X = scaler.transform(df)

    # ======================================================
    # Prediction
    # ======================================================

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    # ======================================================
    # Add Results
    # ======================================================

    result = df.copy()

    result["Fraud Prediction"] = predictions

    result["Fraud Probability"] = probabilities

    result["Fraud Prediction"] = result["Fraud Prediction"].map(

        {

            0: "Legitimate",

            1: "Fraud"

        }

    )

    # ======================================================
    # Summary
    # ======================================================

    total = len(result)

    fraud = (result["Fraud Prediction"] == "Fraud").sum()

    legitimate = total - fraud

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)

    col2.metric("Fraud", fraud)

    col3.metric("Legitimate", legitimate)

    st.markdown("---")

    st.subheader("Prediction Results")

    st.dataframe(result)

    # ======================================================
    # Fraud Distribution
    # ======================================================

    st.subheader("Fraud Distribution")

    chart = result["Fraud Prediction"].value_counts()

    st.bar_chart(chart)

    # ======================================================
    # Download
    # ======================================================

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(

        "Download Predictions",

        csv,

        file_name="fraud_predictions.csv",

        mime="text/csv"

    )

    # ======================================================
    # Workflow
    # ======================================================

    st.markdown("---")

    st.subheader("Prediction Workflow")

    st.code(
        """
User Upload CSV
        ↓
Load Dataset
        ↓
Preprocessing
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Random Forest Model
        ↓
Fraud Prediction
        ↓
Fraud Probability
        ↓
Display Results
        ↓
Download CSV
"""
    )