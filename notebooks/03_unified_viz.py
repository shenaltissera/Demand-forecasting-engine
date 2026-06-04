"""
Unified Forecast Visualizations
Compares LightGBM vs Prophet side-by-side on the merged output.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import warnings, os
warnings.filterwarnings("ignore")

UNIFIED_F = "../outputs/unified_forecasts.parquet"
OUT_DIR   = "../outputs/unified_plots"
os.makedirs(OUT_DIR, exist_ok=True)

df      = pd.read_parquet(UNIFIED_F)
hist    = df[df["Units_Sold"].notna()].copy()
lgbm_df = hist[hist["model_used"] == "lgbm"]
prop_df = hist[hist["model_used"] == "prophet"]

print(f"Loaded {len(df):,} rows | {df['SKU'].nunique()} SKUs")
print(f"  LGBM rows   : {len(lgbm_df):,} ({lgbm_df['SKU'].nunique()} SKUs)")
print(f"  Prophet rows: {len(prop_df):,} ({prop_df['SKU'].nunique()} SKUs)")

# ── 1. Model Coverage Map ─────────────────────────────────────────────────────
model_per_sku = df.groupby("SKU")["model_used"].first().reset_index()
model_per_sku["product"]   = model_per_sku["SKU"].str.split("_").str[0]
model_per_sku["store"]     = model_per_sku["SKU"].str.split("_").str[1]
pivot = model_per_sku.pivot(index="product", columns="store", values="model_used")

fig, ax = plt.subplots(figsize=(10, 8))
cmap = {"lgbm": "#2196F3", "prophet": "#FF9800"}
for i, prod in enumerate(pivot.index):
    for j, store in enumerate(pivot.columns):
        val = pivot.loc[prod, store]
        if pd.notna(val):
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                         color=cmap.get(val, "gray"), alpha=0.85))
            ax.text(j + 0.5, i + 0.5, val[:4].upper(),
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")

ax.set_xlim(0, len(pivot.columns))
ax.set_ylim(0, len(pivot.index))
ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_xlabel("Store"); ax.set_ylabel("Product")
ax.set_title("1 · Model Assignment Map (SKU × Store)", fontsize=13, fontweight="bold")
n_lgbm   = lgbm_df["SKU"].nunique()
n_prophet = prop_df["SKU"].nunique()
lgbm_patch   = mpatches.Patch(color="#2196F3", label=f"LightGBM ({n_lgbm} SKUs)")
prophet_patch = mpatches.Patch(color="#FF9800", label=f"Prophet  ({n_prophet} SKUs)")
ax.legend(handles=[lgbm_patch, prophet_patch], loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_model_map.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 01_model_map.png")

# ── 2. Side-by-side MAE comparison ───────────────────────────────────────────
def sku_mae(g):
    return np.mean(np.abs(g["Units_Sold"] - g["forecast"]))

mae_lgbm   = lgbm_df.groupby("SKU").apply(sku_mae).values
mae_prophet = prop_df.groupby("SKU").apply(sku_mae).values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(mae_lgbm,    bins=20, color="#2196F3", edgecolor="white", alpha=0.85, label="LightGBM")
axes[0].hist(mae_prophet, bins=20, color="#FF9800", edgecolor="white", alpha=0.65, label="Prophet")
axes[0].axvline(np.median(mae_lgbm),    color="#2196F3", lw=2, linestyle="--")
axes[0].axvline(np.median(mae_prophet), color="#FF9800", lw=2, linestyle="--")
axes[0].set_title("MAE Distribution per SKU", fontweight="bold")
axes[0].set_xlabel("MAE (units)"); axes[0].set_ylabel("# SKUs")
axes[0].legend()

summary = pd.DataFrame({
    "Model":  ["LightGBM", "Prophet"],
    "Median MAE":  [round(np.median(mae_lgbm),2),   round(np.median(mae_prophet),2)],
    "Mean MAE":    [round(np.mean(mae_lgbm),2),      round(np.mean(mae_prophet),2)],
})
bars = axes[1].bar(summary["Model"], summary["Median MAE"],
                   color=["#2196F3","#FF9800"], width=0.4, alpha=0.9)
for bar, v in zip(bars, summary["Median MAE"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.5,
                 f"{v:.2f}", ha="center", fontweight="bold")
axes[1].set_title("Median MAE by Model", fontweight="bold")
axes[1].set_ylabel("Median MAE (units)")

plt.suptitle("2 · LightGBM vs Prophet — MAE Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_mae_comparison.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 02_mae_comparison.png")

# ── 3. CI Coverage comparison ─────────────────────────────────────────────────
def sku_ci(g):
    return ((g["Units_Sold"] >= g["lower_80"]) & (g["Units_Sold"] <= g["upper_80"])).mean() * 100

ci_lgbm    = lgbm_df.groupby("SKU").apply(sku_ci).values
ci_prophet = prop_df.groupby("SKU").apply(sku_ci).values

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(ci_lgbm,    bins=20, color="#2196F3", edgecolor="white", alpha=0.8, label=f"LightGBM  (mean {np.mean(ci_lgbm):.1f}%)")
ax.hist(ci_prophet, bins=20, color="#FF9800", edgecolor="white", alpha=0.65, label=f"Prophet   (mean {np.mean(ci_prophet):.1f}%)")
ax.axvline(80, color="tomato", lw=2, linestyle="--", label="Target 80%")
ax.set_title("3 · 80% Confidence Interval Coverage — Both Models", fontsize=13, fontweight="bold")
ax.set_xlabel("CI Coverage (%)"); ax.set_ylabel("# SKUs")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_ci_coverage.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 03_ci_coverage.png")

# ── 4. Best-of-both SKU deep dive ─────────────────────────────────────────────
best_lgbm   = lgbm_df.groupby("SKU").apply(sku_mae).idxmin()
best_prophet = prop_df.groupby("SKU").apply(sku_mae).idxmin()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, sku, model, color in [
    (axes[0], best_lgbm,    "LightGBM", "#2196F3"),
    (axes[1], best_prophet, "Prophet",  "#FF9800"),
]:
    g = df[df["SKU"] == sku].sort_values("Date")
    ax.plot(g["Date"], g["Units_Sold"], color="steelblue", lw=1, alpha=0.7, label="Actual")
    ax.plot(g["Date"], g["forecast"],  color=color, lw=1.5, label=f"{model} Forecast")
    ax.fill_between(g["Date"], g["lower_80"], g["upper_80"],
                    alpha=0.15, color=color, label="80% CI")
    mae_val = np.mean(np.abs(g["Units_Sold"].dropna() - g.loc[g["Units_Sold"].notna(), "forecast"]))
    ax.set_title(f"Best {model} SKU: {sku}  |  MAE = {mae_val:.2f} units",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Units Sold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

plt.suptitle("4 · Best-Performing SKU per Model", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_best_sku_per_model.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 04_best_sku_per_model.png")

# ── 5. Daily forecast vs actual (aggregated) ─────────────────────────────────
daily = hist.groupby("Date").agg(
    actual=("Units_Sold", "sum"),
    forecast=("forecast", "sum"),
    lower=("lower_80", "sum"),
    upper=("upper_80", "sum"),
).reset_index()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily["Date"], daily["actual"],   color="steelblue", lw=1,   alpha=0.6, label="Total Actual Demand")
ax.plot(daily["Date"], daily["forecast"], color="tomato",    lw=1.5, label="Total Unified Forecast")
ax.fill_between(daily["Date"], daily["lower"], daily["upper"],
                alpha=0.12, color="tomato", label="Aggregated 80% CI")
ax.set_title("5 · Portfolio-Level: Actual vs Unified Forecast (All 100 SKUs)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Total Units Sold / Forecast")
ax.set_xlabel("Date")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_portfolio_forecast.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 05_portfolio_forecast.png")

print(f"\n✅ All plots saved to {OUT_DIR}/")
