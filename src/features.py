"""Feature engineering pipeline for F1 race prediction."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML features from raw race results.

    Features:
    - grid_position: starting grid position
    - driver_avg_finish: rolling avg finish position (last 5 races)
    - constructor_avg_finish: rolling avg finish for constructor (last 5 races)
    - circuit_driver_avg: driver's historical avg finish at this circuit
    - driver_dnf_rate: proportion of DNFs in last 10 races
    - grid_position_change: driver's avg positions gained/lost from grid
    """
    df = df.copy()
    df = df.dropna(subset=["finish_position"])
    df["finish_position"] = df["finish_position"].astype(int)
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    # Mark DNFs (status is "DNF", "DNS", or "Disqualified" from OpenF1)
    df["is_dnf"] = (~df["status"].isin(["Finished"])).astype(int)

    # Rolling driver average finish (last 5 races)
    df["driver_avg_finish"] = (
        df.groupby("driver_id")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Rolling constructor average finish (last 5 races)
    df["constructor_avg_finish"] = (
        df.groupby("constructor_id")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Driver's historical average at this circuit
    circuit_driver_stats = (
        df.groupby(["circuit_id", "driver_id"])["finish_position"]
        .expanding().mean().reset_index(level=[0, 1], drop=True)
    )
    df["circuit_driver_avg"] = circuit_driver_stats.shift(1)

    # Driver DNF rate (last 10 races)
    df["driver_dnf_rate"] = (
        df.groupby("driver_id")["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Average grid position change (positions gained/lost)
    df["pos_change"] = df["grid_position"] - df["finish_position"]
    df["grid_position_change"] = (
        df.groupby("driver_id")["pos_change"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Fill NaN features with reasonable defaults
    feature_cols = [
        "grid_position",
        "driver_avg_finish",
        "constructor_avg_finish",
        "circuit_driver_avg",
        "driver_dnf_rate",
        "grid_position_change",
    ]
    df["circuit_driver_avg"] = df["circuit_driver_avg"].fillna(df["driver_avg_finish"])
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df, feature_cols
