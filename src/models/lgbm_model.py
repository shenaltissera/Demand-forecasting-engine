import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import pickle, os

FEATURE_COLS = [
    # Existing business forecast — most powerful signal
    "Demand_Forecast",
    # Calendar
    "day_of_week", "week_of_year", "month", "quarter", "is_weekend",
    # Demand history
    "sales_lag_7d", "sales_lag_14d", "sales_lag_28d",
    "rolling_mean_7d", "rolling_std_7d",
    "rolling_mean_28d", "rolling_std_28d",
    # Pricing & promotions
    "Price", "Discount", "Competitor_Pricing",
    "Holiday_Promotion", "promo_lag_7d",
    # Encoded categoricals
    "Weather_Condition_enc", "Seasonality_enc",
    "Category_enc", "Region_enc",
]
TARGET    = "Units_Sold"
MODEL_PATH = "outputs/lgbm_model.pkl"


def train(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X, y = df[available], df[TARGET]

    params = {
        "objective":        "regression_l1",   # MAE — robust to demand spikes
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          -1,
    }

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []

    print("\n── Time-Series Cross Validation ──")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=500)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val).clip(0)
        mape  = np.mean(np.abs((y_val - preds) / (y_val + 1))) * 100
        scores.append(mape)
        print(f"  Fold {fold+1}/5 — MAPE: {mape:.2f}%")

    print(f"\n  ✓ Mean CV MAPE : {np.mean(scores):.2f}%")
    print(f"  ✓ Best fold    : {min(scores):.2f}%")

    # Final model on full data
    final = lgb.LGBMRegressor(**params, n_estimators=500)
    final.fit(X, y)

    os.makedirs("outputs", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((final, available), f)
    print(f"\n✓ Model saved → {MODEL_PATH}")
    return final, available


def predict(df: pd.DataFrame) -> pd.DataFrame:
    with open(MODEL_PATH, "rb") as f:
        model, available = pickle.load(f)

    preds        = model.predict(df[available]).clip(0)
    std_estimate = preds * 0.25
    out          = df[["SKU", "Date", TARGET]].copy()
    out["forecast"]  = preds
    out["lower_80"]  = (preds - 1.28 * std_estimate).clip(0)
    out["upper_80"]  =  preds + 1.28 * std_estimate
    return out


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    with open(MODEL_PATH, "rb") as f:
        model, available = pickle.load(f)
    imp = pd.DataFrame({
        "feature":    available,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/features.parquet")
    train(df)
