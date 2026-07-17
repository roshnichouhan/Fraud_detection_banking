import joblib
import pandas as pd


model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict(input_dataframe):

    data = scaler.transform(input_dataframe)

    prediction = model.predict(data)

    return prediction


if __name__ == "__main__":

    sample = pd.DataFrame([
        {
            "amount": 2500,
            "transaction_hour": 14,
            "merchant_risk": 1
        }
    ])

    result = predict(sample)

    print(result)