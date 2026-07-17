import os
import pandas as pd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "nlp_transactions.csv"
)

df = pd.read_csv(DATA_PATH)

print(df.head())

df["transaction_time"] = pd.to_datetime(
    df["transaction_time"]
)


df["transaction_hour"] = df["transaction_time"].dt.hour

df["day_of_week"] = df["transaction_time"].dt.dayofweek

df["month"] = df["transaction_time"].dt.month

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["high_value_transaction"] = (
    df["transaction_amount"] > 5000
).astype(int)

df["international_transaction"] = (
    df["is_international"]
)

merchant_freq = df["merchant"].value_counts()

df["merchant_frequency"] = (
    df["merchant"].map(merchant_freq)
)


customer_freq = (
    df.groupby("customer_id")
      .size()
)

df["customer_transaction_count"] = (
    df["customer_id"].map(customer_freq)
)

df["risk_score"] = (
      df["previous_fraud_count"] * 2
    + df["high_value_transaction"]
    + df["international_transaction"]
)

def amount_category(amount):

    if amount < 500:
        return "Low"

    elif amount < 5000:
        return "Medium"

    else:
        return "High"


df["amount_category"] = (
    df["transaction_amount"]
      .apply(amount_category)
)

df = pd.get_dummies(
    df,
    columns=["amount_category"],
    drop_first=True
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "engineered_data.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Feature Engineering Completed Successfully!")

