import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================================
# Project Paths
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_model.pkl"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "scaler.pkl"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "engineered_data.csv"
)

# ==========================================================
# Load Resources
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, scaler


@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


# ==========================================================
# Main Function
# ==========================================================

def show_model_performance():

    st.title("📈 Model Performance Dashboard")

    st.markdown("""
Evaluate the trained Machine Learning model using
multiple performance metrics.
""")

    st.markdown("---")

    # ======================================================
    # Load Files
    # ======================================================

    try:

        model, scaler = load_model()

        df = load_dataset()

    except Exception as e:

        st.error(e)

        return

    # ======================================================
    # Check Target Column
    # ======================================================

    if "label" not in df.columns:

        st.error("'label' column not found.")

        return

    # ======================================================
    # Encode Categorical Columns
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
    # Features & Target
    # ======================================================

    X = df.drop("label", axis=1)

    y = df["label"]

    # ======================================================
    # Train Test Split
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    # ======================================================
    # Scaling
    # ======================================================

    X_test = scaler.transform(X_test)

    # ======================================================
    # Prediction
    # ======================================================

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # ======================================================
    # Metrics
    # ======================================================

    accuracy = accuracy_score(

        y_test,

        y_pred

    )

    precision = precision_score(

        y_test,

        y_pred,

        zero_division=0

    )

    recall = recall_score(

        y_test,

        y_pred,

        zero_division=0

    )

    f1 = f1_score(

        y_test,

        y_pred,

        zero_division=0

    )

    roc_auc = roc_auc_score(

        y_test,

        y_prob

    )

    # ======================================================
    # KPI Cards
    # ======================================================

    st.subheader("📊 Model Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(

            "Accuracy",

            f"{accuracy:.4f}"

        )

    with col2:

        st.metric(

            "Precision",

            f"{precision:.4f}"

        )

    with col3:

        st.metric(

            "Recall",

            f"{recall:.4f}"

        )

    with col4:

        st.metric(

            "F1 Score",

            f"{f1:.4f}"

        )

    with col5:

        st.metric(

            "ROC-AUC",

            f"{roc_auc:.4f}"

        )

    st.markdown("---")

    # ======================================================
    # Metric Description
    # ======================================================

    st.subheader("📖 Metric Explanation")

    st.info(f"""

**Accuracy:** {accuracy:.4f}

Percentage of correctly classified transactions.

---

**Precision:** {precision:.4f}

Out of all transactions predicted as fraud,
how many were actually fraud.

---

**Recall:** {recall:.4f}

Out of all actual fraud transactions,
how many were detected.

---

**F1 Score:** {f1:.4f}

Balances Precision and Recall.

---

**ROC-AUC:** {roc_auc:.4f}

Measures the model's ability to distinguish
between fraud and genuine transactions.

""")

    st.markdown("---")

    st.success("✅ Model Evaluation Completed Successfully")