"""
Model Tests
"""

from api.model_loader import (
    load_artifacts,
    get_model,
    get_scaler,
    get_label_encoders
)


# ----------------------------------------------------
# Load Artifacts
# ----------------------------------------------------

def test_load_artifacts():

    load_artifacts()

    assert get_model() is not None

    assert get_scaler() is not None

    assert get_label_encoders() is not None


# ----------------------------------------------------
# Model Prediction
# ----------------------------------------------------

def test_model_prediction():

    load_artifacts()

    model = get_model()

    scaler = get_scaler()

    label_encoders = get_label_encoders()

    sample = [[
        5000,
        0,
        0,
        1,
        0
    ]]

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    assert prediction is not None

    assert len(prediction) == 1