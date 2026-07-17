import os
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "engineered_data.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(df.head())

# ============================================================
# Encode Categorical Columns
# ============================================================

label_encoders = {}

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    label_encoders[column] = encoder

# ============================================================
# Features & Target
# ============================================================

X = df.drop("label", axis=1)

y = df["label"]

feature_names = X.columns

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================================
# Grid Search
# ============================================================

print("\nRunning GridSearchCV...\n")

param_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [5, 10, None],

    "min_samples_split": [2, 5, 10]

}

grid = GridSearchCV(

    RandomForestClassifier(random_state=42),

    param_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1

)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("=" * 60)
print("Best Parameters")
print("=" * 60)

print(grid.best_params_)

print()

print("Best CV Score :", grid.best_score_)

# ============================================================
# Cross Validation
# ============================================================

scores = cross_val_score(

    best_model,

    X_train,

    y_train,

    cv=5,

    scoring="accuracy"

)

print("\nCross Validation Accuracy")

print(scores)

print("Average :", scores.mean())

# ============================================================
# Prediction
# ============================================================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]

# ============================================================
# Metrics
# ============================================================

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("Precision :", precision_score(y_test, y_pred))

print("Recall :", recall_score(y_test, y_pred))

print("F1 Score :", f1_score(y_test, y_pred))

print("ROC AUC :", roc_auc_score(y_test, y_prob))

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

# ============================================================
# ROC Curve
# ============================================================

fpr, tpr, _ = roc_curve(

    y_test,

    y_prob

)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="ROC Curve")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# ============================================================
# Precision Recall Curve
# ============================================================

precision, recall, _ = precision_recall_curve(

    y_test,

    y_prob

)

plt.figure(figsize=(6,5))

plt.plot(recall, precision)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.show()

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": best_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop Feature Importance\n")

print(importance.head(15))

# ============================================================
# SHAP Explainability
# ============================================================

try:

    import shap

    print("\nGenerating SHAP values...")

    explainer = shap.TreeExplainer(best_model)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(

        shap_values,

        X_test,

        feature_names=feature_names

    )

except Exception:

    print("\nSHAP not installed.")

    print("Install using")

    print("pip install shap")

# ============================================================
# Save Best Model
# ============================================================

joblib.dump(

    best_model,

    os.path.join(

        MODEL_DIR,

        "best_model.pkl"

    )

)

joblib.dump(

    scaler,

    os.path.join(

        MODEL_DIR,

        "scaler.pkl"

    )

)

joblib.dump(

    label_encoders,

    os.path.join(

        MODEL_DIR,

        "label_encoders.pkl"

    )

)

print("\n" + "=" * 60)

print("Best Model Saved Successfully")

print("=" * 60)

print("best_model.pkl")

print("scaler.pkl")

print("label_encoders.pkl")