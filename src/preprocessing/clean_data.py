import pandas as pd
import os

# -----------------------------
# Load Dataset
# -----------------------------
input_file = "data/raw/bank_transactions.csv"

df = pd.read_csv(input_file)

print("Original Shape:", df.shape)

# -----------------------------
# Missing Values
# -----------------------------

# Fill missing transaction_amount with median
df["transaction_amount"] = df["transaction_amount"].fillna(
    df["transaction_amount"].median()
)

# Fill missing merchant names
df["merchant"] = df["merchant"].fillna("Unknown")

# -----------------------------
# Remove Duplicate Rows
# -----------------------------
df = df.drop_duplicates()

print("Cleaned Shape:", df.shape)

# -----------------------------
# Create processed folder
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Save Clean Dataset
# -----------------------------
output_file = "data/processed/clean_transactions.csv"

df.to_csv(output_file, index=False)

print(f"Clean dataset saved successfully at: {output_file}")