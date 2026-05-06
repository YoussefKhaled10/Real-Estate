import pandas as pd
import numpy as np
from category_encoders import TargetEncoder


# ============================================================
# Target Encoding for categorical features
# ============================================================

def target_encode_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    categorical_cols: list,
    min_samples_leaf: int = 5,
    smoothing: int = 10
):
    """
    Apply Target Encoding on categorical columns.

    - Fit encoder on X_train only
    - Transform both X_train and X_test
    - Avoid data leakage

    Returns:
    --------
    X_train_encoded : pd.DataFrame
    X_test_encoded  : pd.DataFrame
    encoder         : fitted TargetEncoder
    """

    encoder = TargetEncoder(
        cols=categorical_cols,
        min_samples_leaf=min_samples_leaf,
        smoothing=smoothing
    )

    # Fit only on training data
    encoder.fit(X_train[categorical_cols], y_train)

    # Transform train and test
    train_encoded = encoder.transform(X_train[categorical_cols])
    test_encoded  = encoder.transform(X_test[categorical_cols])

    return train_encoded, test_encoded, encoder


# ============================================================
# Build final feature matrices
# ============================================================

def build_feature_matrix(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_encoded: pd.DataFrame,
    test_encoded: pd.DataFrame,
    drop_cols: list
):
    """
    Combine encoded categorical features with numeric features.
    """

    # Drop original categorical columns
    X_train_num = X_train.drop(columns=drop_cols)
    X_test_num  = X_test.drop(columns=drop_cols)

    # Concatenate numeric + encoded
    X_train_all = pd.concat([X_train_num, train_encoded], axis=1)
    X_test_all  = pd.concat([X_test_num, test_encoded], axis=1)

    return X_train_all, X_test_all


# ============================================================
# Log transform target
# ============================================================

def log_transform_target(y: pd.Series):
    """
    Apply log1p transformation to target variable.
    """
    return np.log1p(y)