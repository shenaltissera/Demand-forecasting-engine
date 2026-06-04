"""
Prophet Model Visualizations
Generates 4 types of plots for each SKU + a summary dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle, os, warnings
warnings.filterwarnings("ignore")

MODEL_DIR   = "../outputs/prophet_models"
FORECAST_F  = "../outputs/prophet_forecasts.parquet"
ACTUALS_F   = "../data/processed/cleaned.parquet"
OUT_DIR     = "../outputs/prophet_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
forecasts = pd.read_parquet(FORECAST_F)
actuals   = pd.read_parquet(ACTUALS_F)
skus      = forecasts["SKU"].unique()
cutoff    = actuals["Date"].max()   # last date with actual data

print(f"Loaded {len(skus)} Prophet SKUs | cutoff: {cutoff.date()}")

# ── 1. Individual SKU forecast plots (top 6 by volatility) ───────────────────
cv_rank = (
    actuals.groupby("SKU")["Units_Sold"]
    .apply(lambda x: x.std() / (x.mean() + 1))
    .reindex(skus)
    .sort_values(ascending=False)
)
top6 = cv_rank.head(6).index.tolist()

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for i, sku in enumerate(top6):
    ax  = axes[i]
    fc  = forecasts[forecasts["SKU"] == sku].sort_values("Date")
    act = actuals[actuals["SKU"] == sku].sort_values("Date")

    hist_fc  = fc[fc["Date"] <= cutoff]
    fut_fc   = fc[fc["Date"] >  cutoff]

    # Actual demand
    ax.plot(act["Date"], act["Units_Sold"],
            color="steelblue", lw=1, alpha=0.7, label="Actual Demand")

    # Historical forecast + CI
    ax.plot(hist_fc["Date"], hist_fc["forecast"],
            color="tomato", lw=1.5, label="Prophet Forecast")
    ax.fill_between(hist_fc["Date"], hist_fc["lower_80"], hist_fc["upper_80"],
                    alpha=0.15, color="tomato", label="80% CI")

    # Future forecast
    if not fut_fc.empty:
        ax.plot(fut_fc["Date"], fut_fc["forecast"],
                color="darkorange", lw=2, linestyle="--", label="Future (28d)")
        ax.fill_between(fut_fc["Date"], fut_fc["lower_80"], fut_fc["upper_80"],
                        alpha=0.25, color="darkorange")

    # Cutoff line
    ax.axvline(cutoff, color="gray", lw=1, linestyle=":", alpha=0.8)
    ax.text(cutoff, ax.get_ylim()[1] * 0.92, " forecast →",
            fontsize=7, color="gray")

    ax.set_title(f"SKU: {sku}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Units Sold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")

plt.suptitle("Prophet Forecasts — Top 6 Most Volatile SKUs", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_sku_forecasts.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 01_sku_forecasts.png")

# ── 2. Seasonality decomposition (one SKU) ───────────────────────────────────
sample_sku = top6[0]
model_path = os.path.join(MODEL_DIR, f"{sample_sku}.pkl")

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    act_sku = actuals[actuals["SKU"] == sample_sku].sort_values("Date")
    ts = act_sku[["Date", "Units_Sold"]].rename(
        columns={"Date": "ds", "Units_Sold": "y"})
    if "Holiday_Promotion" in act_sku.columns:
        ts["promo"] = act_sku["Holiday_Promotion"].values

    future_df = model.make_future_dataframe(periods=28)
    if "promo" in model.extra_regressors:
        future_df["promo"] = 0
    fc_raw = model.predict(future_df)

    # Extract components
    components = ["trend", "weekly", "yearly"]
    present    = [c for c in components if c in fc_raw.columns]

    fig, axes = plt.subplots(len(present), 1, figsize=(14, 4 * len(present)))
    if len(present) == 1:
        axes = [axes]

    titles = {"trend": "Trend", "weekly": "Weekly Seasonality", "yearly": "Yearly Seasonality"}
    colors = {"trend": "steelblue", "weekly": "darkorange", "yearly": "seagreen"}

    for ax, comp in zip(axes, present):
        ax.plot(fc_raw["ds"], fc_raw[comp], color=colors[comp], lw=1.5)
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_title(titles[comp], fontsize=11, fontweight="bold")
        ax.set_ylabel("Effect (units)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"Seasonality Decomposition — SKU: {sample_sku}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_seasonality_decomposition.png", bbox_inches="tight", dpi=130)
    plt.close()
    print("✓ 02_seasonality_decomposition.png")

# ── 3. Forecast error distribution ───────────────────────────────────────────
merged = forecasts.merge(actuals[["SKU", "Date", "Units_Sold"]],
                         on=["SKU", "Date"], how="inner")
merged = merged[merged["Date"] <= cutoff]
merged["error"]    = merged["forecast"] - merged["Units_Sold"]
merged["abs_error"] = merged["error"].abs()
merged["in_ci"]    = ((merged["Units_Sold"] >= merged["lower_80"]) &
                      (merged["Units_Sold"] <= merged["upper_80"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Error distribution
axes[0].hist(merged["error"].clip(-300, 300), bins=60,
             color="steelblue", edgecolor="white", alpha=0.8)
axes[0].axvline(0, color="tomato", lw=2, linestyle="--")
axes[0].set_title("Forecast Error Distribution", fontweight="bold")
axes[0].set_xlabel("Forecast − Actual (units)")
axes[0].set_ylabel("Count")
mu = merged["error"].mean()
axes[0].text(0.97, 0.95, f"Mean bias: {mu:+.1f}",
             transform=axes[0].transAxes, ha="right", va="top",
             fontsize=9, color="tomato")

# Absolute error by SKU (top 10 worst)
sku_mae = merged.groupby("SKU")["abs_error"].mean().sort_values(ascending=False).head(10)
axes[1].barh(sku_mae.index, sku_mae.values, color="tomato", alpha=0.8)
axes[1].set_title("Top 10 SKUs by MAE", fontweight="bold")
axes[1].set_xlabel("Mean Absolute Error (units)")
axes[1].tick_params(axis="y", labelsize=8)

# CI coverage per SKU
ci_cov = merged.groupby("SKU")["in_ci"].mean() * 100
axes[2].hist(ci_cov, bins=20, color="seagreen", edgecolor="white", alpha=0.8)
axes[2].axvline(80, color="tomato", lw=2, linestyle="--", label="Target 80%")
axes[2].set_title("80% CI Coverage Distribution", fontweight="bold")
axes[2].set_xlabel("Coverage (%)")
axes[2].set_ylabel("# SKUs")
axes[2].legend(fontsize=9)
axes[2].text(0.97, 0.95, f"Mean: {ci_cov.mean():.1f}%",
             transform=axes[2].transAxes, ha="right", va="top", fontsize=9)

plt.suptitle("3 · Prophet Forecast Error Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_error_analysis.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 03_error_analysis.png")

# ── 4. Future 28-day forecast heatmap across all SKUs ────────────────────────
future = forecasts[forecasts["Date"] > cutoff].copy()
pivot  = future.pivot_table(index="SKU", columns="Date",
                            values="forecast", aggfunc="mean")

fig, ax = plt.subplots(figsize=(18, 10))
im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Forecasted Units Sold", shrink=0.6)

ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=7)

date_labels = [d.strftime("%b %d") for d in pivot.columns]
step = max(1, len(date_labels) // 14)
ax.set_xticks(range(0, len(date_labels), step))
ax.set_xticklabels(date_labels[::step], rotation=30, fontsize=8)

ax.set_title("4 · 28-Day Demand Forecast Heatmap — All Prophet SKUs",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("SKU")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_forecast_heatmap.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 04_forecast_heatmap.png")

# ── 5. Actual vs Forecast scatter ─────────────────────────────────────────────
sample = merged.sample(min(5000, len(merged)), random_state=42)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(sample["Units_Sold"], sample["forecast"],
           alpha=0.15, s=8, color="steelblue")
lim = max(sample["Units_Sold"].max(), sample["forecast"].max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect forecast")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Actual Units Sold"); ax.set_ylabel("Forecasted Units")
ax.set_title("5 · Actual vs Forecast (Prophet SKUs)", fontsize=12, fontweight="bold")
ax.legend(); ax.set_aspect("equal")

corr = sample["Units_Sold"].corr(sample["forecast"])
ax.text(0.05, 0.93, f"R = {corr:.3f}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_actual_vs_forecast.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ 05_actual_vs_forecast.png")

print(f"\n✅ All plots saved to {OUT_DIR}/")
print(f"\nSummary stats:")
print(f"  Median MAE     : {merged['abs_error'].groupby(merged['SKU']).mean().median():.2f} units")
print(f"  Mean CI Cover  : {ci_cov.mean():.2f}%")
print(f"  Forecast bias  : {merged['error'].mean():+.2f} units")
print(f"  R (act vs fc)  : {merged['Units_Sold'].corr(merged['forecast']):.3f}")
