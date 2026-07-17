"""
schemas.py

Pydantic models used for request validation.
"""

from typing import List
from pydantic import BaseModel, Field


# ---------------------------------------------------
# Single Prediction Schema
# ---------------------------------------------------

class PredictionRequest(BaseModel):
    transaction_amount: float = Field(
        ...,
        gt=0,
        description="Transaction amount"
    )

    merchant: str = Field(
        ...,
        description="Merchant Name"
    )

    device: str = Field(
        ...,
        description="Device Type"
    )

    previous_fraud_count: int = Field(
        ...,
        ge=0,
        description="Previous Fraud Count"
    )

    international_transaction: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 = Domestic, 1 = International"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_amount": 5000,
                "merchant": "Amazon",
                "device": "Mobile",
                "previous_fraud_count": 2,
                "international_transaction": 1
            }
        }


# ---------------------------------------------------
# Batch Prediction Schema
# ---------------------------------------------------

class BatchPredictionRequest(BaseModel):
    transactions: List[PredictionRequest]

    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_amount": 5000,
                        "merchant": "Amazon",
                        "device": "Mobile",
                        "previous_fraud_count": 2,
                        "international_transaction": 1
                    },
                    {
                        "transaction_amount": 1200,
                        "merchant": "Flipkart",
                        "device": "Desktop",
                        "previous_fraud_count": 0,
                        "international_transaction": 0
                    }
                ]
            }
        }