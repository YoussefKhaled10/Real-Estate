import pandas as pd


def predict(model, X: pd.DataFrame):
    """
    Make predictions using a trained model.

    Parameters
    ----------
    model : trained model
    X     : pd.DataFrame (features)

    Returns
    -------
    predictions : np.ndarray
    """
    return model.predict(X)