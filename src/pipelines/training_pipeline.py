import json
import pandas as pd
from pathlib import Path

from src.models.train import train_lightgbm
from src.models.evaluate import evaluate_regression
from src.models.save_load import save_model


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("../data/processed")
ARTIFACTS_DIR = Path("../artifacts")

TRAIN_PATH = DATA_DIR / "train_encoded.csv"
TEST_PATH = DATA_DIR / "test_encoded.csv"

BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_lgb_params.json"

MODEL_PATH = ARTIFACTS_DIR / "final_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"


# ============================================================
# TRAINING PIPELINE
# ============================================================

def run_training_pipeline():
    """
    End-to-end training pipeline using:
    - Encoded data
    - Best hyperparameters from Optuna
    """

    # --------------------------------------------------------
    # 1) Load encoded data
    # --------------------------------------------------------
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "price_in_USD"

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Test shape : {X_test.shape}")

    # --------------------------------------------------------
    # 2) Load best hyperparameters (from hyper_tuning.ipynb)
    # --------------------------------------------------------
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)

    print("[INFO] Loaded best hyperparameters")

    # --------------------------------------------------------
    # 3) Train final LightGBM model
    # --------------------------------------------------------
    model = train_lightgbm(
        X_train=X_train,
        y_train=y_train,
        params=best_params
    )

    print("[INFO] Model training completed")

    # --------------------------------------------------------
    # 4) Optional evaluation
    # --------------------------------------------------------
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_metrics = evaluate_regression(y_train, train_preds)
    test_metrics = evaluate_regression(y_test, test_preds)

    print("===== FINAL MODEL METRICS =====")
    print("Train:", train_metrics)
    print("Test :", test_metrics)

    # --------------------------------------------------------
    # 5) Save final model
    # --------------------------------------------------------
    save_model(model, MODEL_PATH)
    print(f"[INFO] Model saved -> {MODEL_PATH}")

    # --------------------------------------------------------
    # 6) Save feature order
    # --------------------------------------------------------
    feature_list = list(X_train.columns)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_list, f)

    print(f"[INFO] Feature list saved -> {FEATURES_PATH}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    run_training_pipeline()