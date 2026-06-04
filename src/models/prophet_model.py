import pandas as pd
import numpy as np
from prophet import Prophet
import os, pickle

MODEL_DIR = "outputs/prophet_models"


def train_sku(sku: str, df_sku: pd.DataFrame) -> Prophet:
    """Train a Prophet model for a single SKU."""
    ts = df_sku[["Date", "Units_Sold"]].rename(columns={"Date": "ds", "Units_Sold": "y"})
    ts["y"] = ts["y"].clip(lower=0)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,       # 80% confidence interval
        changepoint_prior_scale=0.1,
    )
    model.fit(ts)
    return model


def forecast_sku(model: Prophet, periods: int = 28) -> pd.DataFrame:
    """Generate future forecast for one SKU."""
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)


def run_all_skus(df: pd.DataFrame, periods: int = 28, max_skus: int = None):
    """
    Run Prophet on slow-moving / intermittent SKUs.
    Use LightGBM for high-volume SKUs instead.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    skus = df["SKU"].unique()
    if max_skus:
        skus = skus[:max_skus]

    results = []
    for i, sku in enumerate(skus):
        df_sku = df[df["SKU"] == sku].sort_values("Date")
        if len(df_sku) < 30:  # skip SKUs with too little history
            continue
        try:
            model = train_sku(sku, df_sku)
            forecast = forecast_sku(model, periods)
            forecast["SKU"] = sku
            forecast = forecast.rename(columns={
                "ds": "Date", "yhat": "forecast",
                "yhat_lower": "lower_80", "yhat_upper": "upper_80"
            })
            results.append(forecast)
        except Exception as e:
            print(f"  Skipping SKU {sku}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(skus)} SKUs")

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/cleaned.parquet")
    forecasts = run_all_skus(df, periods=28, max_skus=50)  # test with 50 SKUs first
    forecasts.to_parquet("outputs/prophet_forecasts.parquet", index=False)
    print(forecasts.head())
