import streamlit as st


def show_about():
    """
    About Page
    AI Banking Fraud Detection System
    """

    st.title("ℹ️ About the Project")

    st.markdown("---")

    # =====================================================
    # Project Overview
    # =====================================================

    st.header("🏦 Project Overview")

    st.write("""
This project is an **AI-powered Banking Fraud Detection System**
designed to identify potentially fraudulent financial transactions
using Machine Learning and Natural Language Processing (NLP).

The application provides an end-to-end fraud detection pipeline,
including data preprocessing, feature engineering, model training,
evaluation, explainability, and real-time prediction through an
interactive Streamlit dashboard.

The primary objective is to help financial institutions detect
fraudulent transactions quickly while reducing false alarms.
""")

    # =====================================================
    # Dataset
    # =====================================================

    st.header("📂 Dataset Description")

    st.write("""
The dataset contains banking transaction records collected for fraud
analysis.

Typical features include:

- Transaction Amount
- Transaction Type
- Merchant Category
- Merchant Risk Score
- Account Balance
- Transaction Time
- Device Type
- Location
- Previous Fraud Count
- Risk Score
- Transaction Description
- Fraud Label (Target Variable)

The target variable is:

- **0 → Legitimate Transaction**
- **1 → Fraudulent Transaction**
""")

    # =====================================================
    # Machine Learning
    # =====================================================

    st.header("🤖 Machine Learning Models")

    st.write("""
The project currently uses:

- Random Forest Classifier
- Hyperparameter Tuning using GridSearchCV
- Cross Validation
- Feature Scaling using StandardScaler
- Label Encoding for categorical variables

Model evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
""")

    # =====================================================
    # NLP
    # =====================================================

    st.header("📝 NLP Pipeline")

    st.write("""
The fraud detection system also processes transaction descriptions.

Pipeline:

1. Text Cleaning
2. Lowercase Conversion
3. Punctuation Removal
4. Stopword Removal
5. Lemmatization
6. TF-IDF Vectorization
7. Feature Generation
8. Integration with Structured Banking Features
""")

    # =====================================================
    # Feature Engineering
    # =====================================================

    st.header("⚙️ Feature Engineering")

    st.write("""
Several engineered features improve model performance:

- Transaction Hour
- Weekend Transaction
- High Value Transaction Flag
- Merchant Frequency
- Customer Transaction Count
- Previous Fraud Count
- Risk Score
- Account Activity Features
- TF-IDF Text Features
""")

    # =====================================================
    # Technologies
    # =====================================================

    st.header("🛠 Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### Programming

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Joblib

### NLP

- TF-IDF
- NLTK
        """)

    with col2:
        st.markdown("""
### Visualization

- Plotly
- Matplotlib

### Dashboard

- Streamlit

### Version Control

- Git
- GitHub
        """)

    # =====================================================
    # Project Workflow
    # =====================================================

    st.header("🔄 Project Workflow")

    st.code("""
Raw Dataset
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
NLP Processing
      ↓
TF-IDF Features
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Evaluation
      ↓
Prediction
      ↓
Streamlit Dashboard
""")

    # =====================================================
    # Key Features
    # =====================================================

    st.header("⭐ Key Features")

    st.markdown("""
- Real-Time Fraud Prediction
- Batch Prediction from CSV
- Interactive Dashboard
- Model Performance Monitoring
- Feature Importance Analysis
- Explainable AI (SHAP Ready)
- Hyperparameter Tuning
- Cross Validation
- Download Prediction Results
""")

    # =====================================================
    # Author
    # =====================================================

    st.header("👨‍💻 Author")

    st.info("""
**AI Banking Fraud Detection System**

Developed as an end-to-end Data Science and Machine Learning project
to demonstrate production-ready ML pipeline development.

Skills Demonstrated:

• Data Cleaning

• Exploratory Data Analysis

• Feature Engineering

• Natural Language Processing

• Machine Learning

• Model Evaluation

• Hyperparameter Tuning

• Explainable AI

• Streamlit Dashboard Development

• End-to-End Deployment
""")

    # =====================================================
    # Footer
    # =====================================================

    st.markdown("---")

    st.success(
        "🚀 Recruiter-Ready Machine Learning Project | Banking Fraud Detection System"
    )