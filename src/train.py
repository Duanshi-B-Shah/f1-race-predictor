"""Train the F1 race position prediction model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import numpy as np
from xgboost import XGBRegressor

from features import build_features

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "race_results.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"


def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df, feature_cols = build_features(df)

    target = "finish_position"
    X = df[feature_cols]
    y = df[target]

    # Time-based split: train on earlier seasons, test on latest
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Baseline: predict finish = grid position
    baseline_preds = X_test["grid_position"]
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    print(f"\nBaseline (grid=finish) MAE: {baseline_mae:.2f}")

    # XGBoost model
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))

    print(f"XGBoost MAE: {mae:.2f}")
    print(f"XGBoost RMSE: {rmse:.2f}")
    print(f"Improvement over baseline: {baseline_mae - mae:.2f} positions")

    # Cross-validation with time series split
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        model_cv = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model_cv.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        cv_preds = model_cv.predict(X.iloc[val_idx])
        cv_scores.append(mean_absolute_error(y.iloc[val_idx], cv_preds))

    print(f"\nCV MAE (5-fold time series): {sum(cv_scores)/len(cv_scores):.2f} "
          f"(± {pd.Series(cv_scores).std():.2f})")

    # Save final model trained on all data
    final_model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    final_model.fit(X, y, verbose=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")
    print(f"\nModel saved to {MODEL_DIR}/xgb_model.pkl")


if __name__ == "__main__":
    train()
