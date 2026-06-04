import pandas as pd
import numpy as np


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error — ignores zero-demand days."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def stockout_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """% of days where forecast was BELOW actual demand (under-forecast = stockout risk)."""
    return (y_pred < y_true).mean() * 100


def coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """% of actuals that fall within the confidence interval."""
    return ((y_true >= lower) & (y_true <= upper)).mean() * 100


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate forecasts per SKU.
    Expects columns: SKU, Units_Sold, forecast, lower_80, upper_80
    """
    results = []
    for sku, group in df.groupby("SKU"):
        results.append({
            "SKU": sku,
            "MAPE": mape(group["Units_Sold"], group["forecast"]),
            "Stockout_Rate": stockout_rate(group["Units_Sold"], group["forecast"]),
            "CI_Coverage_80": coverage(group["Units_Sold"], group["lower_80"], group["upper_80"]),
            "n_days": len(group),
        })

    report = pd.DataFrame(results)
    print(f"\n=== Evaluation Summary ===")
    print(f"Median MAPE:        {report['MAPE'].median():.2f}%")
    print(f"Median Stockout %:  {report['Stockout_Rate'].median():.2f}%")
    print(f"Mean CI Coverage:   {report['CI_Coverage_80'].mean():.2f}%")
    return report


if __name__ == "__main__":
    # Example usage after generating forecasts
    df = pd.read_parquet("outputs/lgbm_forecasts.parquet")
    report = evaluate(df)
    report.to_csv("outputs/evaluation_report.csv", index=False)
