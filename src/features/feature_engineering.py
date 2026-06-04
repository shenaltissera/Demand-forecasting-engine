import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from Date."""
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [7, 14, 28]) -> pd.DataFrame:
    """Add lagged demand features per SKU (mimics Zara's weekly replenishment cycle)."""
    df = df.sort_values(["SKU", "Date"])
    for lag in lags:
        df[f"sales_lag_{lag}d"] = df.groupby("SKU")["Units_Sold"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = [7, 28]) -> pd.DataFrame:
    """Rolling mean and std demand per SKU."""
    df = df.sort_values(["SKU", "Date"])
    for w in windows:
        rolled = df.groupby("SKU")["Units_Sold"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_mean_{w}d"] = rolled
        rolled_std = df.groupby("SKU")["Units_Sold"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        )
        df[f"rolling_std_{w}d"] = rolled_std.fillna(0)
    return df


def add_stockout_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag days where inventory hit zero (stockout event)."""
    if "Inventory_Level" in df.columns:
        df["stockout"] = (df["Inventory_Level"] == 0).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_stockout_flag(df)
    df = df.dropna(subset=["sales_lag_7d"])  # drop rows without enough history
    return df


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/cleaned.parquet")
    df = build_features(df)
    df.to_parquet("data/processed/features.parquet", index=False)
    print(f"Feature matrix: {df.shape}")
