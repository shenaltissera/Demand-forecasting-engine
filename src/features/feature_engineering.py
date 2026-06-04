import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features from Date."""
    df["day_of_week"]  = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["Date"].dt.month
    df["quarter"]      = df["Date"].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [7, 14, 28]) -> pd.DataFrame:
    """Lagged demand per SKU — mimics Zara's weekly replenishment signal."""
    df = df.sort_values(["SKU", "Date"])
    for lag in lags:
        df[f"sales_lag_{lag}d"] = df.groupby("SKU")["Units_Sold"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = [7, 28]) -> pd.DataFrame:
    """Rolling mean & std demand per SKU."""
    df = df.sort_values(["SKU", "Date"])
    for w in windows:
        df[f"rolling_mean_{w}d"] = df.groupby("SKU")["Units_Sold"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}d"] = df.groupby("SKU")["Units_Sold"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        ).fillna(0)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical features."""
    for col in ["Weather_Condition", "Seasonality", "Category", "Region"]:
        if col in df.columns:
            df[col + "_enc"] = df[col].astype("category").cat.codes
    return df


def add_promo_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Was there a promotion in the last 7 days? (demand echo effect)"""
    if "Holiday_Promotion" in df.columns:
        df["promo_lag_7d"] = df.groupby("SKU")["Holiday_Promotion"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).max()
        )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = encode_categoricals(df)
    df = add_promo_lag(df)
    df = df.dropna(subset=["sales_lag_7d"])
    print(f"✓ Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/cleaned.parquet")
    df = build_features(df)
    df.to_parquet("data/processed/features.parquet", index=False)
