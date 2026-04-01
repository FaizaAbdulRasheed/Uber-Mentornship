"""
End-to-end pipeline: generate → preprocess → engineer → train.
Designed to be called once and cached in Streamlit session_state.
"""

import pandas as pd

from src.data.generate import generate_traffic_data, preprocess
from src.features.engineer import build_features, get_X_y, FEATURE_COLS
from src.models.train import train, load_model, get_feature_importance


def run_pipeline(model_type: str = "gbr", force_retrain: bool = False):
    """
    Run (or load) the full ML pipeline.

    Returns
    -------
    dict with keys: df_raw, df_feat, pipe, metrics,
                    y_test, y_pred, feature_importance
    """
    # --- Data ---
    df_raw = generate_traffic_data(n_days=365, junctions=4)
    df_raw = preprocess(df_raw)

    # --- Features ---
    df_feat = build_features(df_raw)
    X, y = get_X_y(df_feat)

    # --- Model ---
    pipe = None if force_retrain else load_model()
    if pipe is None:
        pipe, metrics, X_test, y_test, y_pred = train(X, y, model_type)
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        y_pred = np.clip(pipe.predict(X_test), 0, None)
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

    fi = get_feature_importance(pipe, FEATURE_COLS)

    return {
        "df_raw":              df_raw,
        "df_feat":             df_feat,
        "pipe":                pipe,
        "metrics":             metrics,
        "y_test":              y_test,
        "y_pred":              y_pred,
        "feature_importance":  fi,
    }
