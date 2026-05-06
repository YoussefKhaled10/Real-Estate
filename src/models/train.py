import lightgbm as lgb
import pandas as pd

# ============================================================
# Train LightGBM Regressor
# ============================================================

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None
):
    """
    Train LightGBM regression model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (log-transformed).
    params : dict, optional
        LightGBM hyperparameters.

    Returns
    -------
    model : trained LightGBM model
    """

    # Default parameters (best baseline)
    if params is None:
        params = {
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5,
            "reg_alpha": 3,
            "random_state": 42
        }

    model = lgb.LGBMRegressor(**params)

    model.fit(X_train, y_train)

    return model
