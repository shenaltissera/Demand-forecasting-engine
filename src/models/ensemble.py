"""
Unified Forecast Table
Merges LightGBM + Prophet outputs into one forecast table.

Routing:
  - SKUs handled by Prophet  → use Prophet forecast
  - SKUs handled by LightGBM → use LightGBM forecast
  - Both columns kept for comparison / blending
"""

import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.prophet_model import classify_skus

LGBM_F     = "outputs/lgbm_forecasts.parquet"
PROPHET_F  = "outputs/prophet_forecasts.parquet"
ACTUALS_F  = "data/processed/cleaned.parquet"
OUT_F      = "outputs/unified_forecasts.parquet"
REPORT_F   = "outputs/unified_evaluation.csv"


# ── Load all sources ──────────────────────────────────────────────────────────

def load_all():
    actuals  = pd.read_parquet(ACTUALS_F)
    lgbm_fc  = pd.read_parquet(LGBM_F)
    prop_fc  = pd.read_parquet(PROPHET_F)

    routing  = classify_skus(actuals)
    return actuals, lgbm_fc, prop_fc, routing


# ── Build unified table ───────────────────────────────────────────────────────

def build_unified(actuals, lgbm_fc, prop_fc, routing) -> pd.DataFrame:
    """
    One row per (SKU, Date).
    Columns: SKU, Date, Units_Sold, model_used,
             forecast, lower_80, upper_80,
             lgbm_forecast, prophet_forecast
    """

    # ── Prophet side ──────────────────────────────────────────────────────────
    prop = prop_fc[["SKU", "Date", "forecast", "lower_80", "upper_80"]].copy()
    prop = prop.rename(columns={
        "forecast": "prophet_forecast",
        "lower_80": "prophet_lower_80",
        "upper_80": "prophet_upper_80",
    })

    # ── LightGBM side ─────────────────────────────────────────────────────────
    lgbm = lgbm_fc[["SKU", "Date", "lgbm_forecast", "lower_80", "upper_80"]].copy()
    lgbm = lgbm.rename(columns={
        "lower_80": "lgbm_lower_80",
        "upper_80": "lgbm_upper_80",
    })

    # ── Actuals (source of truth for Units_Sold) ──────────────────────────────
    acts = actuals[["SKU", "Date", "Units_Sold"]].copy()

    # ── Merge: actuals + lgbm + prophet ───────────────────────────────────────
    df = acts.merge(lgbm, on=["SKU", "Date"], how="left")
    df = df.merge(prop,  on=["SKU", "Date"], how="left")

    # ── Assign active model per SKU ───────────────────────────────────────────
    df["model_used"] = df["SKU"].apply(
        lambda s: "prophet" if s in routing["prophet"] else "lgbm"
    )

    # ── Pick winning forecast & CI ────────────────────────────────────────────
    df["forecast"] = np.where(
        df["model_used"] == "prophet",
        df["prophet_forecast"],
        df["lgbm_forecast"],
    )
    df["lower_80"] = np.where(
        df["model_used"] == "prophet",
        df["prophet_lower_80"],
        df["lgbm_lower_80"],
    )
    df["upper_80"] = np.where(
        df["model_used"] == "prophet",
        df["prophet_upper_80"],
        df["lgbm_upper_80"],
    )

    # ── Clip negatives ────────────────────────────────────────────────────────
    for col in ["forecast", "lower_80", "upper_80",
                "lgbm_forecast", "prophet_forecast"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    return df


# ── Evaluate unified forecasts ────────────────────────────────────────────────

def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    hist = df[df["Units_Sold"].notna()].copy()
    results = []

    for (sku, model), g in hist.groupby(["SKU", "model_used"]):
        y    = g["Units_Sold"]
        yhat = g["forecast"]
        mask = y > 0
        mape = np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100 if mask.sum() else np.nan
        mae  = np.mean(np.abs(y - yhat))
        cov  = ((y >= g["lower_80"]) & (y <= g["upper_80"])).mean() * 100
        stockout = (yhat < y).mean() * 100
        results.append({
            "SKU": sku, "model": model,
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
            "CI_Coverage_80": round(cov, 2),
            "Stockout_Rate": round(stockout, 2),
        })

    report = pd.DataFrame(results)

    print("\n── Unified Forecast Evaluation ────────────────────────")
    for model in ["lgbm", "prophet"]:
        g = report[report["model"] == model]
        print(f"\n  {model.upper()} ({len(g)} SKUs)")
        print(f"    Median MAE       : {g['MAE'].median():.2f} units")
        print(f"    Median MAPE      : {g['MAPE'].median():.2f}%")
        print(f"    Mean CI Coverage : {g['CI_Coverage_80'].mean():.2f}%")
        print(f"    Mean Stockout %  : {g['Stockout_Rate'].mean():.2f}%")

    all_mae  = report['MAE'].median()
    all_mape = report['MAPE'].median()
    print(f"\n  OVERALL ({len(report)} SKUs)")
    print(f"    Median MAE  : {all_mae:.2f} units")
    print(f"    Median MAPE : {all_mape:.2f}%")
    print("────────────────────────────────────────────────────────")
    return report


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    actuals, lgbm_fc, prop_fc, routing = load_all()

    print("\nBuilding unified forecast table...")
    unified = build_unified(actuals, lgbm_fc, prop_fc, routing)

    os.makedirs("outputs", exist_ok=True)
    unified.to_parquet(OUT_F, index=False)

    total    = len(unified)
    future   = unified["Units_Sold"].isna().sum()
    skus     = unified["SKU"].nunique()
    lgbm_n   = (unified["model_used"] == "lgbm").sum()
    prop_n   = (unified["model_used"] == "prophet").sum()

    print(f"\n✓ Unified forecast saved → {OUT_F}")
    print(f"  Total rows    : {total:,}")
    print(f"  Historical    : {total - future:,}")
    print(f"  Future (28d)  : {future:,}")
    print(f"  SKUs          : {skus}")
    print(f"  LightGBM rows : {lgbm_n:,}")
    print(f"  Prophet rows  : {prop_n:,}")

    report = evaluate(unified)
    report.to_csv(REPORT_F, index=False)
    print(f"\n✓ Evaluation report saved → {REPORT_F}")

    print("\nSample rows:")
    print(unified[["SKU","Date","Units_Sold","model_used",
                   "forecast","lower_80","upper_80"]].head(10).to_string(index=False))
