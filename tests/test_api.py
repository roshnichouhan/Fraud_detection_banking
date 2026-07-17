"""
API Tests
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ----------------------------------------------------
# Root Endpoint
# ----------------------------------------------------

def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "Running"


# ----------------------------------------------------
# API Home
# ----------------------------------------------------

def test_api_home():
    response = client.get("/api/")

    assert response.status_code == 200


# ----------------------------------------------------
# Single Prediction
# ----------------------------------------------------

def test_prediction():

    payload = {
        "transaction_amount": 5000,
        "merchant": "Amazon",
        "device": "Mobile",
        "previous_fraud_count": 1,
        "international_transaction": 0
    }

    response = client.post(
        "/api/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "fraud" in data


# ----------------------------------------------------
# Batch Prediction
# ----------------------------------------------------

def test_batch_prediction():

    payload = {
        "transactions": [
            {
                "transaction_amount": 2000,
                "merchant": "Amazon",
                "device": "Mobile",
                "previous_fraud_count": 0,
                "international_transaction": 0
            },
            {
                "transaction_amount": 7000,
                "merchant": "Flipkart",
                "device": "Desktop",
                "previous_fraud_count": 2,
                "international_transaction": 1
            }
        ]
    }

    response = client.post(
        "/api/batch_predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "results" in data


# ----------------------------------------------------
# Invalid Request
# ----------------------------------------------------

def test_invalid_request():

    payload = {
        "merchant": "Amazon"
    }

    response = client.post(
        "/api/predict",
        json=payload
    )

    assert response.status_code == 422