from pathlib import Path
import joblib

# Fraud_detection_banking/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoders.pkl"

model = None
scaler = None
label_encoders = None


def load_artifacts():
    global model, scaler, label_encoders

    # Check files
    for file in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    if model is None:
        print("Loading Machine Learning Model...")
        model = joblib.load(MODEL_PATH)

    if scaler is None:
        print("Loading Scaler...")
        scaler = joblib.load(SCALER_PATH)

    if label_encoders is None:
        print("Loading Label Encoders...")
        label_encoders = joblib.load(ENCODER_PATH)

    print("All artifacts loaded successfully.")


def get_model():
    return model


def get_scaler():
    return scaler


def get_label_encoders():
    return label_encoders