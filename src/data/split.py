import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_and_save(df: pd.DataFrame, target: str, test_size: float = 0.2, save_path: str = "../data/processed/"):
    """
    Split the dataframe into train and test, then save both CSV files.
    Recommended to call this before any imputation or encoding to avoid data leakage.
    """
    os.makedirs(save_path, exist_ok=True)

    if target not in df.columns:
        raise ValueError(f"[ERROR] Target column '{target}' not found in dataframe.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(save_path, "train.csv")
    test_path  = os.path.join(save_path, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] Train saved => {train_path}  shape={train_df.shape}")
    print(f"[INFO] Test saved => {test_path}   shape={test_df.shape}")

    return train_df, test_df