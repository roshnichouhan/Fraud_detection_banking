# ==========================================================
# Banking Fraud Detection Project
# File: src/nlp/vectorizer.py
# Purpose: Convert transaction descriptions into TF-IDF features
# ==========================================================

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================================
# Project Paths
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==========================================================
# Load Dataset Automatically
# ==========================================================

possible_files = [
    "nlp_cleaned_transactions.csv",
    "clean_transactions.csv",
    "cleaned_transactions.csv",
    "processed_transactions.csv",
    "cleaned_data.csv",
    "bank_transactions.csv"
]

input_file = None

for file in possible_files:
    path = os.path.join(PROCESSED_DIR, file)

    if os.path.exists(path):
        input_file = path
        break

if input_file is None:
    raise FileNotFoundError(
        f"No cleaned dataset found inside:\n{PROCESSED_DIR}"
    )

print("=" * 50)
print("Loading Dataset")
print(input_file)
print("=" * 50)

df = pd.read_csv(input_file)

# ==========================================================
# Check Required Column
# ==========================================================

if "transaction_description" not in df.columns:
    raise ValueError(
        "'transaction_description' column not found."
    )

# Fill missing values
df["transaction_description"] = (
    df["transaction_description"]
    .fillna("")
    .astype(str)
)

# ==========================================================
# TF-IDF Vectorization
# ==========================================================

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    df["transaction_description"]
)

# ==========================================================
# Convert to DataFrame
# ==========================================================

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("TF-IDF Shape :", tfidf_df.shape)

# ==========================================================
# Save TF-IDF Features
# ==========================================================

tfidf_output = os.path.join(
    PROCESSED_DIR,
    "tfidf_features.csv"
)

tfidf_df.to_csv(
    tfidf_output,
    index=False
)

print("TF-IDF Features Saved")

# ==========================================================
# Merge Original Dataset + TF-IDF
# ==========================================================

final_df = pd.concat(
    [df.reset_index(drop=True),
     tfidf_df.reset_index(drop=True)],
    axis=1
)

# ==========================================================
# Save Final Dataset
# ==========================================================

final_output = os.path.join(
    PROCESSED_DIR,
    "nlp_transactions.csv"
)

final_df.to_csv(
    final_output,
    index=False
)

print("Final Dataset Saved")

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 50)
print("NLP Pipeline Completed Successfully")
print("=" * 50)

print("Original Shape :", df.shape)
print("TF-IDF Shape   :", tfidf_df.shape)
print("Final Shape    :", final_df.shape)

print("\nFiles Created:")
print("1. tfidf_features.csv")
print("2. nlp_transactions.csv")