import joblib
import os


def save_model(model, path: str):
    """
    Save trained model to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """
    Load trained model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")

    return joblib.load(path)