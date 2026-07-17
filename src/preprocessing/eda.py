import os
import pandas as pd
import matplotlib.pyplot as plt

# ====================================
# Load Dataset
# ====================================

file_path = "data/processed/clean_transactions.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found: {file_path}\n"
        "Run clean_data.py first."
    )

df = pd.read_csv(file_path)

# ====================================
# Basic Information
# ====================================

print("=" * 50)
print("Dataset Shape")
print("=" * 50)
print(df.shape)

print("\n" + "=" * 50)
print("Dataset Information")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("Statistical Summary")
print("=" * 50)
print(df.describe(include="all"))

print("\n" + "=" * 50)
print("Column Names")
print("=" * 50)
print(df.columns.tolist())

print("\n" + "=" * 50)
print("Missing Values")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("Duplicate Rows")
print("=" * 50)
print(df.duplicated().sum())

# ====================================
# Label Distribution
# ====================================

if "label" in df.columns:

    print("\n" + "=" * 50)
    print("Label Counts")
    print("=" * 50)

    print(df["label"].value_counts())

    print("\nFraud Percentage")

    fraud_percentage = df["label"].value_counts(normalize=True) * 100

    print(fraud_percentage)

else:
    print("\n'label' column not found.")

# ====================================
# Transaction Amount Distribution
# ====================================

if "transaction_amount" in df.columns:

    plt.figure(figsize=(8,5))

    plt.hist(df["transaction_amount"], bins=30)

    plt.title("Transaction Amount Distribution")

    plt.xlabel("Transaction Amount")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.show()

else:
    print("\n'transaction_amount' column not found.")

# ====================================
# Boxplot
# ====================================

if "transaction_amount" in df.columns:

    plt.figure(figsize=(6,4))

    plt.boxplot(df["transaction_amount"])

    plt.title("Transaction Amount Outliers")

    plt.show()

# ====================================
# Correlation Matrix
# ====================================

numeric_df = df.select_dtypes(include=["number"])

if not numeric_df.empty:

    corr = numeric_df.corr()

    print("\n" + "=" * 50)
    print("Correlation Matrix")
    print("=" * 50)

    print(corr)

else:
    print("No numeric columns available.")