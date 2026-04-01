"""
Feature engineering for traffic volume forecasting.
Builds lag features, rolling statistics, and temporal encodings.
"""

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_event",
    "temperature", "humidity", "wind", "precipitation",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_3", "rolling_mean_24",
    "rolling_std_3", "rolling_std_24",
    "hour_sin", "hour_cos",
]

TARGET_COL = "vehicles"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features, rolling statistics, and cyclic encodings.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: datetime, junction, vehicles, hour,
        day_of_week, month, is_weekend, is_event, temperature,
        humidity, wind, precipitation.

    Returns
    -------
    pd.DataFrame
        DataFrame with new feature columns appended.
    """
    df = df.copy().sort_values(["junction", "datetime"])

    grp = df.groupby("junction")["vehicles"]

    # Lag features
    df["lag_1"]   = grp.shift(1)
    df["lag_24"]  = grp.shift(24)
    df["lag_168"] = grp.shift(168)  # 1 week

    # Rolling statistics
    df["rolling_mean_3"]  = grp.transform(lambda x: x.shift(1).rolling(3,  min_periods=1).mean())
    df["rolling_mean_24"] = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
    df["rolling_std_3"]   = grp.transform(lambda x: x.shift(1).rolling(3,  min_periods=1).std().fillna(0))
    df["rolling_std_24"]  = grp.transform(lambda x: x.shift(1).rolling(24, min_periods=1).std().fillna(0))

    # Cyclic hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df.dropna(subset=["lag_168"]).reset_index(drop=True)


def get_X_y(df: pd.DataFrame):
    """Return feature matrix X and target vector y."""
    return df[FEATURE_COLS], df[TARGET_COL]
