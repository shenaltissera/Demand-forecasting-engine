"""
Airflow DAG Visualization
Renders the pipeline as a styled flow diagram.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Node definitions ──────────────────────────────────────────────────────────
nodes = {
    "start":            {"pos": (0.5, 9.2),  "color": "#4CAF50", "shape": "circle",  "label": "START",            "sub": ""},
    "ingest_data":      {"pos": (0.5, 7.8),  "color": "#1565C0", "shape": "box",     "label": "ingest_data",      "sub": "Load & validate CSV\nsave cleaned.parquet"},
    "build_features":   {"pos": (0.5, 6.2),  "color": "#1565C0", "shape": "box",     "label": "build_features",   "sub": "Lag · Rolling · Calendar\nCategorical encoding"},
    "train_lgbm":       {"pos": (-1.1, 4.4), "color": "#2196F3", "shape": "box",     "label": "train_lgbm",       "sub": "LightGBM · 5-fold CV\nlgbm_forecasts.parquet"},
    "run_prophet":      {"pos": (2.1, 4.4),  "color": "#FF9800", "shape": "box",     "label": "run_prophet",      "sub": "Prophet · 55 SKUs\nprophet_forecasts.parquet"},
    "merge_forecasts":  {"pos": (0.5, 2.8),  "color": "#6A1B9A", "shape": "box",     "label": "merge_forecasts",  "sub": "Unified forecast table\n100 SKUs · 73k rows"},
    "evaluate":         {"pos": (0.5, 1.3),  "color": "#00695C", "shape": "box",     "label": "evaluate_forecasts","sub": "MAE · MAPE · CI Coverage\nStockout Rate per SKU"},
    "summary":          {"pos": (0.5, -0.1), "color": "#00695C", "shape": "box",     "label": "pipeline_summary", "sub": "XCom metrics report\nRun summary printed"},
    "end":              {"pos": (0.5, -1.4), "color": "#F44336", "shape": "circle",  "label": "END",              "sub": ""},
}

# ── Edges ─────────────────────────────────────────────────────────────────────
edges = [
    ("start",          "ingest_data",     "straight"),
    ("ingest_data",    "build_features",  "straight"),
    ("build_features", "train_lgbm",      "left"),
    ("build_features", "run_prophet",     "right"),
    ("train_lgbm",     "merge_forecasts", "left"),
    ("run_prophet",    "merge_forecasts", "right"),
    ("merge_forecasts","evaluate",        "straight"),
    ("evaluate",       "summary",         "straight"),
    ("summary",        "end",             "straight"),
]

# ── Metrics sidebar ───────────────────────────────────────────────────────────
metrics = [
    ("Schedule",       "Every Monday 06:00"),
    ("Max active runs","1 (no overlap)"),
    ("Task timeout",   "2 hours each"),
    ("Retries",        "2 × 10 min delay"),
    (""),
    ("LightGBM MAE",   "6.36 units"),
    ("LightGBM MAPE",  "16.08%"),
    ("Prophet CI Cov", "83.77%"),
    (""),
    ("Total SKUs",     "100"),
    ("  └ LightGBM",   "45 SKUs"),
    ("  └ Prophet",    "55 SKUs"),
    (""),
    ("Unified rows",   "73,100"),
    ("Future (28d)",   "1,925 rows"),
]

fig, ax = plt.subplots(figsize=(14, 13))
ax.set_xlim(-3.2, 5.5)
ax.set_ylim(-2.2, 10.2)
ax.axis("off")
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

# ── Draw edges ────────────────────────────────────────────────────────────────
def get_edge_points(src, tgt, style):
    sx, sy = nodes[src]["pos"]
    tx, ty = nodes[tgt]["pos"]
    return (sx, sy - 0.28), (tx, ty + 0.28)

for src, tgt, style in edges:
    (sx, sy), (tx, ty) = get_edge_points(src, tgt, style)
    ax.annotate("",
        xy=(tx, ty), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#555E6B",
            lw=2,
            connectionstyle="arc3,rad=0.0" if style == "straight"
                else ("arc3,rad=0.25" if style == "right" else "arc3,rad=-0.25"),
        )
    )

# ── Draw nodes ────────────────────────────────────────────────────────────────
for name, n in nodes.items():
    x, y   = n["pos"]
    color  = n["color"]
    label  = n["label"]
    sub    = n["sub"]
    shape  = n["shape"]

    if shape == "circle":
        circle = plt.Circle((x, y), 0.28, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=6)
    else:
        w, h = 2.2, 0.75
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.06",
                             facecolor=color, edgecolor="white",
                             linewidth=1.2, alpha=0.92, zorder=5)
        ax.add_patch(box)
        ax.text(x, y + 0.13, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=6)
        if sub:
            ax.text(x, y - 0.18, sub, ha="center", va="center",
                    fontsize=7, color="#BBDEFB", zorder=6,
                    fontstyle="italic")

# ── Parallel badge ────────────────────────────────────────────────────────────
ax.text(0.5, 3.65, "⟵  runs in parallel  ⟶",
        ha="center", va="center", fontsize=8.5,
        color="#FFD54F", style="italic")
ax.plot([-1.1, 2.1], [3.72, 3.72], color="#FFD54F", lw=0.8, linestyle="--", alpha=0.5)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_x = 3.8
ax.text(sidebar_x, 9.8, "Pipeline Config", ha="left", va="center",
        fontsize=10, fontweight="bold", color="white")
ax.plot([sidebar_x - 0.1, sidebar_x + 1.5], [9.55, 9.55], color="#444", lw=1)

y_pos = 9.2
for item in metrics:
    if isinstance(item, tuple) and len(item) == 2:
        k, v = item
        ax.text(sidebar_x,      y_pos, k, ha="left", va="center",
                fontsize=7.5, color="#9E9E9E")
        ax.text(sidebar_x + 1.6, y_pos, v, ha="right", va="center",
                fontsize=7.5, color="#E0E0E0", fontweight="bold")
        y_pos -= 0.48
    else:
        y_pos -= 0.22

# ── XCom badge ───────────────────────────────────────────────────────────────
xcom_box = FancyBboxPatch((3.65, 3.0), 1.6, 1.9,
                           boxstyle="round,pad=0.08",
                           facecolor="#1A237E", edgecolor="#3F51B5",
                           linewidth=1.2, alpha=0.9, zorder=4)
ax.add_patch(xcom_box)
ax.text(4.45, 4.72, "XCom Keys", ha="center", fontsize=8,
        fontweight="bold", color="#7986CB")
xcom_keys = ["n_rows · n_skus", "feature_shape",
             "lgbm_skus · prophet_skus",
             "unified_rows · future_rows",
             "lgbm_median_mae",
             "lgbm_median_mape",
             "prop_ci_coverage"]
for i, k in enumerate(xcom_keys):
    ax.text(4.45, 4.42 - i * 0.33, k, ha="center", fontsize=6.8,
            color="#C5CAE9")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color="#1565C0", label="Core pipeline task"),
    mpatches.Patch(color="#2196F3", label="LightGBM model"),
    mpatches.Patch(color="#FF9800", label="Prophet model"),
    mpatches.Patch(color="#6A1B9A", label="Ensemble merge"),
    mpatches.Patch(color="#00695C", label="Evaluation"),
]
ax.legend(handles=legend_items, loc="lower left",
          facecolor="#161B22", edgecolor="#30363D",
          labelcolor="white", fontsize=8, framealpha=0.9)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(0.5, 10.05,
        "Demand Forecasting Pipeline  ·  Airflow DAG",
        ha="center", va="center", fontsize=13, fontweight="bold", color="white")
ax.text(0.5, 9.72,
        "demand_forecast_pipeline  ·  schedule: 0 6 * * 1  ·  max_active_runs: 1",
        ha="center", va="center", fontsize=8, color="#9E9E9E")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/airflow_dag.png", bbox_inches="tight",
            dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f"✅ DAG diagram saved → {OUT_DIR}/airflow_dag.png")
