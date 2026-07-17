import streamlit as st


def show_home():

    # ==========================================================
    # Title
    # ==========================================================

    st.title("🏦 AI Banking Fraud Detection System")

    st.markdown(
        """
Welcome to the **AI-powered Banking Fraud Detection Dashboard**.

This project demonstrates an end-to-end Machine Learning pipeline
for detecting fraudulent banking transactions using
Artificial Intelligence and Natural Language Processing.
"""
    )

    st.markdown("---")

    # ==========================================================
    # KPI Cards
    # ==========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            "50,000+"
        )

    with col2:
        st.metric(
            "Fraud Cases",
            "2,150"
        )

    with col3:
        st.metric(
            "Detection Accuracy",
            "98.4%"
        )

    with col4:
        st.metric(
            "ML Model",
            "Random Forest"
        )

    st.markdown("---")

    # ==========================================================
    # Project Overview
    # ==========================================================

    st.header("📌 Project Overview")

    st.write(
        """
The AI Banking Fraud Detection System predicts whether a banking
transaction is **Fraudulent** or **Legitimate**.

The application combines:

- Machine Learning
- Feature Engineering
- Natural Language Processing
- Explainable AI
- Interactive Streamlit Dashboard

The goal is to assist financial institutions in identifying
high-risk transactions before financial losses occur.
"""
    )

    # ==========================================================
    # Dataset Summary
    # ==========================================================

    st.header("📂 Dataset Summary")

    st.write(
        """
The dataset contains historical banking transactions.

Example features include:

• Transaction Amount

• Merchant Category

• Device Type

• Transaction Description

• Previous Fraud Count

• Risk Score

• International Transaction

• Transaction Time

Target Variable:

**Label**

0 → Genuine

1 → Fraud
"""
    )

    st.markdown("---")

    # ==========================================================
    # Fraud Detection Workflow
    # ==========================================================

    st.header("🔄 Fraud Detection Workflow")

    st.code(
"""
Raw Banking Transactions
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Text Cleaning
        ↓
TF-IDF Feature Extraction
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
Random Forest Model
        ↓
Hyperparameter Tuning
        ↓
Fraud Prediction
        ↓
Interactive Dashboard
"""
    )

    # ==========================================================
    # Key Features
    # ==========================================================

    st.header("⭐ Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("✔ Single Transaction Prediction")

        st.success("✔ Batch Prediction")

        st.success("✔ Fraud Probability")

        st.success("✔ Feature Engineering")

        st.success("✔ NLP Pipeline")

    with col2:

        st.success("✔ Model Performance")

        st.success("✔ ROC Curve")

        st.success("✔ Feature Importance")

        st.success("✔ SHAP Explainability")

        st.success("✔ Download Prediction Results")

    st.markdown("---")

    # ==========================================================
    # Technology Stack
    # ==========================================================

    st.header("🛠 Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.subheader("Programming")

        st.write("""
- Python
- Pandas
- NumPy
""")

    with tech2:

        st.subheader("Machine Learning")

        st.write("""
- Scikit-Learn
- Random Forest
- GridSearchCV
- StandardScaler
""")

    with tech3:

        st.subheader("Visualization")

        st.write("""
- Streamlit
- Plotly
- Matplotlib
""")

    st.markdown("---")

    # ==========================================================
    # Why This Project
    # ==========================================================

    st.header("💼 Why This Project Matters")

    st.info(
        """
This project demonstrates a complete Machine Learning workflow
similar to what is used in production environments.

Skills demonstrated:

• Data Cleaning

• Exploratory Data Analysis

• Feature Engineering

• Natural Language Processing

• Machine Learning

• Hyperparameter Tuning

• Cross Validation

• Explainable AI

• Streamlit Dashboard Development

• Model Deployment
"""
    )

    st.markdown("---")

    # ==========================================================
    # Footer
    # ==========================================================

    st.success(
        "🚀 Recruiter-Ready End-to-End Machine Learning Project"
    )