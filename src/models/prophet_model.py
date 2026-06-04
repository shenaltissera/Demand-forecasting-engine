"""
Prophet model for slow-moving / high-volatility SKUs.

Routing logic (mirrors real Zara use case):
  - CV (coeff of variation) > 0.82  →  Prophet  (volatile, hard for LGBM)
  - zero_demand_pct > 0.5%          →  Prophet  (intermittent)
  - everything else                 →  LightGBM

Prophet strengths here:
  - Handles irregular demand spikes
  - Built-in 80% confidence intervals
  - Captures weekly + yearly seasonality automatically
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import pickle, os, warnings
warnings.filterwarnings("ignore")

MODEL_DIR   = "outputs/prophet_models"
FORECAST_OUT = "outputs/prophet_forecasts.parquet"


# ── SKU routing ───────────────────────────────────────────────────────────────

def classify_skus(df: pd.DataFrame, cv_threshold: float = 0.82,
                  zero_pct_threshold: float = 0.005) -> dict:
    """
    Return dict: {'prophet': [...skus], 'lgbm': [...skus]}
    """
    stats = df.groupby("SKU")["Units_Sold"].agg(
        mean="mean",
        cv=lambda x: x.std() / (x.mean() + 1),
        zero_pct=lambda x: (x == 0).mean(),
    )
    prophet_mask = (stats["cv"] > cv_threshold) | (stats["zero_pct"] > zero_pct_threshold)
    prophet_skus = stats[prophet_mask].index.tolist()
    lgbm_skus    = stats[~prophet_mask].index.tolist()

    print(f"✓ SKU routing:")
    print(f"   Prophet  (volatile/intermittent) : {len(prophet_skus):>4} SKUs")
    print(f"   LightGBM (high-volume, stable)   : {len(lgbm_skus):>4} SKUs")
    return {"prophet": prophet_skus, "lgbm": lgbm_skus}


# ── Prophet training ──────────────────────────────────────────────────────────

def _build_model(use_regressors: bool = True) -> Prophet:
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,            # 80% CI
        changepoint_prior_scale=0.15,   # slightly flexible trend
        seasonality_prior_scale=10,
        seasonality_mode="multiplicative",
    )


def train_one_sku(sku: str, df_sku: pd.DataFrame) -> Prophet:
    """Fit Prophet on a single SKU time-series."""
    ts = (
        df_sku[["Date", "Units_Sold"]]
        .rename(columns={"Date": "ds", "Units_Sold": "y"})
        .sort_values("ds")
    )
    ts["y"] = ts["y"].clip(lower=0)

    # Add promotion as a regressor if available
    if "Holiday_Promotion" in df_sku.columns:
        ts["promo"] = df_sku["Holiday_Promotion"].values
        model = _build_model()
        model.add_regressor("promo")
    else:
        model = _build_model()

    model.fit(ts, iter=300)
    return model


def forecast_one_sku(model: Prophet, df_sku: pd.DataFrame,
                     periods: int = 28) -> pd.DataFrame:
    """Generate forecast for existing dates + `periods` future days."""
    has_promo = "promo" in model.extra_regressors

    # History dates
    history = df_sku[["Date"]].rename(columns={"Date": "ds"}).copy()
    if has_promo:
        history["promo"] = df_sku["Holiday_Promotion"].values

    # Future dates
    last_date = df_sku["Date"].max()
    future_dates = pd.DataFrame({
        "ds": pd.date_range(last_date + pd.Timedelta(days=1), periods=periods)
    })
    if has_promo:
        future_dates["promo"] = 0   # assume no promo in future by default

    future = pd.concat([history, future_dates], ignore_index=True)
    fc     = model.predict(future)

    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={
        "ds": "Date", "yhat": "forecast",
        "yhat_lower": "lower_80", "yhat_upper": "upper_80"
    })


# ── Run all Prophet SKUs ──────────────────────────────────────────────────────

def run_prophet_pipeline(df: pd.DataFrame, periods: int = 28,
                         save_models: bool = True) -> pd.DataFrame:
    """
    Full Prophet pipeline:
      1. Route SKUs
      2. Train + forecast each Prophet SKU
      3. Return combined forecast DataFrame
    """
    routing = classify_skus(df)
    prophet_skus = routing["prophet"]

    if not prophet_skus:
        print("⚠ No SKUs routed to Prophet — all handled by LightGBM.")
        return pd.DataFrame()

    os.makedirs(MODEL_DIR, exist_ok=True)
    results = []
    failed  = []

    print(f"\nTraining Prophet on {len(prophet_skus)} SKUs...")
    for i, sku in enumerate(prophet_skus, 1):
        df_sku = df[df["SKU"] == sku].sort_values("Date").copy()

        if len(df_sku) < 60:
            print(f"  [{i:>3}/{len(prophet_skus)}] SKIP {sku} — not enough history ({len(df_sku)} days)")
            continue

        try:
            model = train_one_sku(sku, df_sku)
            fc    = forecast_one_sku(model, df_sku, periods=periods)
            fc["SKU"] = sku

            # Clip negatives
            fc["forecast"]  = fc["forecast"].clip(lower=0)
            fc["lower_80"]  = fc["lower_80"].clip(lower=0)
            fc["upper_80"]  = fc["upper_80"].clip(lower=0)

            results.append(fc)

            if save_models:
                model_path = os.path.join(MODEL_DIR, f"{sku}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            print(f"  [{i:>3}/{len(prophet_skus)}] ✓ {sku}")

        except Exception as e:
            print(f"  [{i:>3}/{len(prophet_skus)}] ✗ {sku} — {e}")
            failed.append(sku)

    if not results:
        print("⚠ No forecasts generated.")
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    combined.to_parquet(FORECAST_OUT, index=False)
    print(f"\n✓ Prophet forecasts saved → {FORECAST_OUT}")
    print(f"  {len(results)} SKUs succeeded | {len(failed)} failed")
    print(f"  Rows: {len(combined):,}  |  Future rows (28d): {(combined['Date'] > df['Date'].max()).sum():,}")
    return combined


# ── Evaluate Prophet vs actuals ───────────────────────────────────────────────

def evaluate_prophet(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Compare Prophet forecasts against actuals for historical dates."""
    merged = forecasts.merge(
        actuals[["SKU", "Date", "Units_Sold"]],
        on=["SKU", "Date"], how="inner"
    )
    if merged.empty:
        print("⚠ No overlapping dates to evaluate.")
        return pd.DataFrame()

    results = []
    for sku, g in merged.groupby("SKU"):
        y, yhat = g["Units_Sold"], g["forecast"]
        mask = y > 0
        mape = np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100 if mask.sum() > 0 else np.nan
        mae  = np.mean(np.abs(y - yhat))
        cov  = ((y >= g["lower_80"]) & (y <= g["upper_80"])).mean() * 100
        results.append({"SKU": sku, "MAE": round(mae, 2),
                         "MAPE": round(mape, 2), "CI_Coverage_80": round(cov, 2)})

    report = pd.DataFrame(results)
    print("\n── Prophet Evaluation ─────────────────")
    print(f"  Median MAE        : {report['MAE'].median():.2f} units")
    print(f"  Median MAPE       : {report['MAPE'].median():.2f}%")
    print(f"  Mean CI Coverage  : {report['CI_Coverage_80'].mean():.2f}%")
    print("───────────────────────────────────────")
    return report


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/cleaned.parquet")
    forecasts = run_prophet_pipeline(df, periods=28)

    if not forecasts.empty:
        report = evaluate_prophet(forecasts, df)
        report.to_csv("outputs/prophet_evaluation.csv", index=False)
