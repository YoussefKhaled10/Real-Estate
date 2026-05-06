import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "final_model.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "target_encoder.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"

# ============================================================
# Load artifacts once
# ============================================================

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_ORDER = json.load(f)

# ============================================================
# Inference function
# ============================================================

def predict_price(input_data: dict) -> float:
    """
    Run inference on a single property.

    Parameters
    ----------
    input_data : dict
        Dictionary containing property features.

    Returns
    -------
    float
        Predicted price in USD.
    """

    # Create DataFrame from input
    df = pd.DataFrame([input_data])

    # ------------------------------------
    # Encode categorical features
    # ------------------------------------
    encoded = encoder.transform(df[["country", "location"]])

    # Drop original categorical columns
    df = df.drop(columns=["country", "location"])

    # Combine numerical + encoded features
    X = pd.concat([df, encoded], axis=1)

    # Ensure correct feature order
    X = X[FEATURE_ORDER]

    # ------------------------------------
    # Predict (log space)
    # ------------------------------------
    pred_log = model.predict(X)[0]

    # Inverse log transform
    pred_price = float(np.expm1(pred_log))

    return pred_price