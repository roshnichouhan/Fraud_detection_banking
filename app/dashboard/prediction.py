import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("models/best_model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    return model, scaler


# ==========================================================
# Prediction Page
# ==========================================================

def show_prediction():

    st.title("🔍 Fraud Prediction")

    st.markdown("""
Predict whether a banking transaction is fraudulent
using the trained Machine Learning model.
""")

    st.markdown("---")

    # ======================================================
    # User Inputs
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        previous_fraud_count = st.number_input(
            "Previous Fraud Count",
            min_value=0,
            value=0
        )

        merchant_risk = st.slider(
            "Merchant Risk Score",
            min_value=0,
            max_value=10,
            value=5
        )

    with col2:

        international_transaction = st.selectbox(
            "International Transaction",
            ["No", "Yes"]
        )

        device_type = st.selectbox(
            "Device Type",
            [
                "Mobile",
                "Desktop",
                "Tablet"
            ]
        )

        transaction_hour = st.slider(
            "Transaction Hour",
            0,
            23,
            12
        )

    # ======================================================
    # Feature Engineering
    # ======================================================

    international_flag = 1 if international_transaction == "Yes" else 0

    device_mapping = {
        "Mobile": 0,
        "Desktop": 1,
        "Tablet": 2
    }

    device_encoded = device_mapping[device_type]

    # ======================================================
    # Predict Button
    # ======================================================

    if st.button("🚀 Predict Fraud"):

        try:

            model, scaler = load_artifacts()

            input_df = pd.DataFrame([
                {
                    "transaction_amount": amount,
                    "previous_fraud_count": previous_fraud_count,
                    "merchant_risk": merchant_risk,
                    "international_transaction": international_flag,
                    "device_type": device_encoded,
                    "transaction_hour": transaction_hour
                }
            ])

            # Scale

            input_scaled = scaler.transform(
                input_df
            )

            # Prediction

            prediction = model.predict(
                input_scaled
            )[0]

            probability = model.predict_proba(
                input_scaled
            )[0][1]

            st.markdown("---")

            st.subheader("Prediction Result")

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            # ==================================================
            # Fraud / Genuine
            # ==================================================

            if prediction == 1:

                st.error(
                    f"🚨 FRAUD DETECTED\n\nConfidence: {probability*100:.2f}%"
                )

            else:

                st.success(
                    f"✅ GENUINE TRANSACTION\n\nConfidence: {(1-probability)*100:.2f}%"
                )

            # ==================================================
            # Risk Meter
            # ==================================================

            st.progress(
                float(probability)
            )

            # ==================================================
            # Recommendation
            # ==================================================

            st.subheader("Recommendation")

            if probability >= 0.80:

                st.error(
                    """
High Risk Transaction

Recommended Action:

• Block transaction

• Verify customer identity

• Trigger fraud investigation
"""
                )

            elif probability >= 0.50:

                st.warning(
                    """
Medium Risk Transaction

Recommended Action:

• Additional verification

• OTP confirmation

• Monitor account activity
"""
                )

            else:

                st.success(
                    """
Low Risk Transaction

Recommended Action:

• Approve transaction

• Continue monitoring
"""
                )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

    st.markdown("---")

    # ======================================================
    # Prediction Workflow
    # ======================================================

    st.subheader("🔄 Prediction Workflow")

    st.code("""
User Input
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Scaler
      ↓
Best Model
      ↓
Prediction
      ↓
Display Result
""")

    st.markdown("---")

    st.info(
        """
Model: Random Forest

Features Used:

• Transaction Amount

• Previous Fraud Count

• Merchant Risk

• International Transaction

• Device Type

• Transaction Hour
"""
    )