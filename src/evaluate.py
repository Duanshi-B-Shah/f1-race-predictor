"""Evaluate model performance and generate visualizations."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

from features import build_features

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "race_results.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def evaluate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df, feature_cols = build_features(df)

    model = joblib.load(MODEL_DIR / "xgb_model.pkl")

    # Use last 20% as test set (time-based)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    X_test = test_df[feature_cols]
    y_test = test_df["finish_position"]

    test_df["predicted"] = model.predict(X_test)
    test_df["predicted_rounded"] = test_df["predicted"].round().astype(int).clip(1, 20)

    # 1. Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    importance.plot.barh(ax=ax, color="#e10600")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    print("Saved feature_importance.png")

    # 2. Predicted vs Actual scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, test_df["predicted"], alpha=0.3, s=10, color="#1e41ff")
    ax.plot([1, 20], [1, 20], "r--", linewidth=1)
    ax.set_xlabel("Actual Finish Position")
    ax.set_ylabel("Predicted Finish Position")
    ax.set_title("Predicted vs Actual Finish Position")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=150)
    print("Saved predicted_vs_actual.png")

    # 3. Error distribution
    errors = test_df["predicted"] - y_test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=30, color="#00d2be", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Prediction Error (positions)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (MAE={mean_absolute_error(y_test, test_df['predicted']):.2f})")
    ax.axvline(0, color="red", linestyle="--")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_distribution.png", dpi=150)
    print("Saved error_distribution.png")

    # 4. MAE by grid position
    test_df["grid_bucket"] = pd.cut(test_df["grid_position"], bins=[0, 3, 10, 20], labels=["Top 3", "4-10", "11-20"])
    mae_by_grid = test_df.groupby("grid_bucket", observed=True).apply(
        lambda g: mean_absolute_error(g["finish_position"], g["predicted"])
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    mae_by_grid.plot.bar(ax=ax, color=["#e10600", "#1e41ff", "#00d2be"], edgecolor="black")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by Grid Position Group")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mae_by_grid.png", dpi=150)
    print("Saved mae_by_grid.png")

    plt.close("all")
    print("\nAll evaluation outputs saved to outputs/")


if __name__ == "__main__":
    evaluate()
