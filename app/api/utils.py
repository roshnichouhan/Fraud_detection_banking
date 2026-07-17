"""
utils.py

Utility functions for preprocessing input data
before making predictions.
"""

import pandas as pd


def encode_categorical_features(df, label_encoders):
    """
    Encode categorical columns using saved LabelEncoders.
    """

    categorical_columns = [
        "merchant",
        "device"
    ]

    for column in categorical_columns:

        if column in df.columns and column in label_encoders:

            encoder = label_encoders[column]

            # Handle unseen categories
            df[column] = df[column].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )

            df[column] = encoder.transform(df[column])

    return df


def scale_features(df, scaler):
    """
    Scale features using the trained scaler.
    """

    scaled_data = scaler.transform(df)

    return scaled_data


def preprocess_data(df, scaler, label_encoders):
    """
    Complete preprocessing pipeline.
    """

    df = encode_categorical_features(df, label_encoders)

    df = scale_features(df, scaler)

    return df


def prepare_single_input(request):
    """
    Convert a PredictionRequest into a DataFrame.
    """

    return pd.DataFrame([request.model_dump()])


def prepare_batch_input(request):
    """
    Convert a BatchPredictionRequest into a DataFrame.
    """

    records = [item.model_dump() for item in request.transactions]

    return pd.DataFrame(records)


def format_prediction(prediction, probability=None):
    """
    Format a single prediction response.
    """

    response = {
        "prediction": int(prediction),
        "fraud": bool(prediction)
    }

    if probability is not None:
        response["probability"] = round(float(probability), 4)

    return response


def format_batch_predictions(predictions, probabilities=None):
    """
    Format batch prediction response.
    """

    results = []

    if probabilities is not None:

        for pred, prob in zip(predictions, probabilities):

            results.append(
                format_prediction(
                    pred,
                    prob[1]
                )
            )

    else:

        for pred in predictions:

            results.append(
                format_prediction(pred)
            )

    return {
        "total_transactions": len(results),
        "results": results
    }