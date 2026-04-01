"""
Model training and evaluation for traffic forecasting.
No joblib persistence — model is always trained fresh in-session via
Streamlit's @st.cache_resource, which is safe across all OS/Python versions.
"""

import numpy as np
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_pipeline(model_type: str = "gbr") -> Pipeline:
    """Return a sklearn Pipeline with scaler + regressor."""
    if model_type == "gbr":
        reg = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )
    else:
        reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", reg)])


def train(X, y, model_type: str = "gbr", test_size: float = 0.2):
    """
    Train the pipeline and return (pipeline, metrics_dict, X_test, y_test, y_pred).
    Never saves to disk — caller caches the returned pipeline in memory.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    pipe = build_pipeline(model_type)
    pipe.fit(X_train, y_train)
    y_pred = np.clip(pipe.predict(X_test), 0, None)

    # Naive baseline: repeat last known value
    baseline_pred = np.roll(y_test.values, 1)
    baseline_pred[0] = y_test.values[0]
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    metrics = {
        "mae":          round(mean_absolute_error(y_test, y_pred), 4),
        "rmse":         round(mean_squared_error(y_test, y_pred) ** 0.5, 4),
        "r2":           round(r2_score(y_test, y_pred), 4),
        "baseline_mae": round(baseline_mae, 4),
        "improvement":  round((1 - mean_absolute_error(y_test, y_pred) / baseline_mae) * 100, 1),
    }

    return pipe, metrics, X_test, y_test.values, y_pred


def load_model():
    """Always returns None — model is trained fresh each session via cache."""
    return None


def get_feature_importance(pipe, feature_names):
    """Extract feature importances from the inner estimator."""
    try:
        imp = pipe.named_steps["model"].feature_importances_
        return dict(zip(feature_names, imp))
    except AttributeError:
        return {}
