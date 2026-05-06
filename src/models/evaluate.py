import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression model performance.

    Returns a dictionary with:
    - R2
    - RMSE
    - MSE
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "r2": r2,
        "rmse": rmse,
        "mse": mse
    }