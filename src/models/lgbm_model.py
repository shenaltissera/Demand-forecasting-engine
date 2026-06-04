import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import pickle, os

FEATURE_COLS = [
    "day_of_week", "week_of_year", "month", "quarter", "is_weekend",
    "sales_lag_7d", "sales_lag_14d", "sales_lag_28d",
    "rolling_mean_7d", "rolling_std_7d",
    "rolling_mean_28d", "rolling_std_28d",
    "Price",
]
TARGET = "Units_Sold"
MODEL_PATH = "outputs/lgbm_model.pkl"


def train(df: pd.DataFrame):
    """Train LightGBM with time-series cross-validation."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df[TARGET]

    params = {
        "objective": "regression_l1",   # MAE — robust to demand spikes
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=500)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val).clip(0)
        mape = np.mean(np.abs((y_val - preds) / (y_val + 1))) * 100
        scores.append(mape)
        print(f"  Fold {fold+1} MAPE: {mape:.2f}%")

    print(f"\nMean CV MAPE: {np.mean(scores):.2f}%")

    # Final model on all data
    final_model = lgb.LGBMRegressor(**params, n_estimators=500)
    final_model.fit(X, y)

    os.makedirs("outputs", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Model saved to {MODEL_PATH}")
    return final_model


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Generate point forecasts + confidence intervals."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    available = [c for c in FEATURE_COLS if c in df.columns]
    preds = model.predict(df[available]).clip(0)

    # Approximate 80% confidence interval using residual std
    std_estimate = preds * 0.25  # placeholder — replace with quantile regression
    df = df.copy()
    df["forecast"] = preds
    df["lower_80"] = (preds - 1.28 * std_estimate).clip(0)
    df["upper_80"] = preds + 1.28 * std_estimate
    return df[["SKU", "Date", "forecast", "lower_80", "upper_80"]]


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/features.parquet")
    train(df)
