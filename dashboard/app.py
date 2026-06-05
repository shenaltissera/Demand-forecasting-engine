"""
Zara Demand Forecasting Dashboard
==================================
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zara Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card {
        background: #1E2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }
    .stMetric label { color: #9E9E9E !important; font-size: 13px !important; }
    .stMetric div[data-testid="metric-container"] > div { color: white !important; }
    div[data-testid="stSidebar"] { background-color: #161B22; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

@st.cache_data
def load_data():
    unified  = pd.read_parquet(f"{ROOT}/outputs/unified_forecasts.parquet")
    actuals  = pd.read_parquet(f"{ROOT}/data/processed/cleaned.parquet")
    eval_df  = pd.read_csv(f"{ROOT}/outputs/unified_evaluation.csv")
    return unified, actuals, eval_df

try:
    unified, actuals, eval_df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Run the pipeline first: `python src/ingestion/load_data.py` → `python src/features/feature_engineering.py` → train models")
    data_loaded = False
    st.stop()

hist    = unified[unified["Units_Sold"].notna()].copy()
future  = unified[unified["Units_Sold"].isna()].copy()
cutoff  = actuals["Date"].max()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Zara_Logo.svg/320px-Zara_Logo.svg.png",
             width=120)
    st.markdown("## 📦 Demand Forecasting")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔍 SKU Explorer",
        "📊 Model Comparison",
        "🗓️ 28-Day Forecast",
        "⚠️ Stockout Risk",
    ])

    st.markdown("---")
    st.markdown("**Filters**")
    all_skus      = sorted(unified["SKU"].unique())
    all_models    = ["All", "lgbm", "prophet"]
    all_categories = ["All"] + sorted(actuals["Category"].unique().tolist())

    sel_model    = st.selectbox("Model", all_models)
    sel_category = st.selectbox("Category", all_categories)

    st.markdown("---")
    st.caption(f"📅 Data: Jan 2022 – Jan 2024\n\n🔄 Pipeline: Every Monday 06:00\n\n💾 {len(unified):,} rows · {unified['SKU'].nunique()} SKUs")

# ── Filter helpers ─────────────────────────────────────────────────────────────
def filter_skus(df):
    if sel_model != "All":
        df = df[df["model_used"] == sel_model]
    if sel_category != "All":
        skus_in_cat = actuals[actuals["Category"] == sel_category]["SKU"].unique()
        df = df[df["SKU"].isin(skus_in_cat)]
    return df

hist_f   = filter_skus(hist)
unified_f = filter_skus(unified)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📦 Zara Demand Forecasting Engine")
    st.caption("Probabilistic time-series forecasting across 100 SKU×Store combinations")
    st.markdown("---")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    lgbm_r  = eval_df[eval_df["model"] == "lgbm"]
    prop_r  = eval_df[eval_df["model"] == "prophet"]

    c1.metric("Total SKUs",        f"{unified['SKU'].nunique()}")
    c2.metric("LightGBM MAE",      f"{lgbm_r['MAE'].median():.2f} units",   delta="vs baseline −24%")
    c3.metric("LightGBM MAPE",     f"{lgbm_r['MAPE'].median():.1f}%",       delta="vs baseline −33%")
    c4.metric("Prophet CI Cover",  f"{prop_r['CI_Coverage_80'].mean():.1f}%", delta="target: 80%")
    c5.metric("Forecast Horizon",  "28 days")

    st.markdown("---")

    # ── Portfolio demand over time ─────────────────────────────────────────────
    st.subheader("📈 Portfolio-Level Demand vs Forecast")
    daily = hist_f.groupby("Date").agg(
        actual=("Units_Sold","sum"),
        forecast=("forecast","sum"),
        lower=("lower_80","sum"),
        upper=("upper_80","sum"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["upper"],
        fill=None, mode="lines", line_color="rgba(255,152,0,0)",
        showlegend=False, name="upper"))
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["lower"],
        fill="tonexty", mode="lines", line_color="rgba(255,152,0,0)",
        fillcolor="rgba(255,152,0,0.12)", name="80% CI"))
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["forecast"],
        mode="lines", name="Unified Forecast",
        line=dict(color="#FF9800", width=2)))
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["actual"],
        mode="lines", name="Actual Demand",
        line=dict(color="#2196F3", width=1.5, dash="dot")))
    fig.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Date", yaxis_title="Total Units Sold")
    st.plotly_chart(fig, use_container_width=True)

    # ── Model split + Category demand ─────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Model Assignment")
        model_counts = unified.groupby("model_used")["SKU"].nunique().reset_index()
        model_counts.columns = ["Model","SKUs"]
        fig2 = px.pie(model_counts, names="Model", values="SKUs",
                      color="Model",
                      color_discrete_map={"lgbm":"#2196F3","prophet":"#FF9800"},
                      hole=0.5)
        fig2.update_layout(template="plotly_dark", height=280,
                           margin=dict(l=0,r=0,t=10,b=0),
                           showlegend=True)
        fig2.update_traces(textinfo="label+percent")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("🏷️ Demand by Category")
        cat_df = actuals.merge(hist[["SKU","Date","forecast"]], on=["SKU","Date"], how="left")
        cat_agg = cat_df.groupby("Category").agg(
            actual=("Units_Sold","mean"),
            forecast=("forecast","mean"),
        ).reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Actual",   x=cat_agg["Category"], y=cat_agg["actual"],   marker_color="#2196F3"))
        fig3.add_trace(go.Bar(name="Forecast", x=cat_agg["Category"], y=cat_agg["forecast"], marker_color="#FF9800", opacity=0.8))
        fig3.update_layout(template="plotly_dark", height=280, barmode="group",
                           margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_title="Avg Units Sold")
        st.plotly_chart(fig3, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SKU EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SKU Explorer":
    st.title("🔍 SKU Explorer")
    st.caption("Drill into any individual SKU forecast")
    st.markdown("---")

    col1, col2 = st.columns([1, 3])
    with col1:
        sel_sku = st.selectbox("Select SKU", all_skus)
        sku_model = unified[unified["SKU"] == sel_sku]["model_used"].iloc[0]
        st.info(f"Model: **{sku_model.upper()}**")

        sku_eval = eval_df[eval_df["SKU"] == sel_sku]
        if not sku_eval.empty:
            st.metric("MAE",          f"{sku_eval['MAE'].values[0]:.2f} units")
            st.metric("MAPE",         f"{sku_eval['MAPE'].values[0]:.1f}%")
            st.metric("CI Coverage",  f"{sku_eval['CI_Coverage_80'].values[0]:.1f}%")
            st.metric("Stockout Rate",f"{sku_eval['Stockout_Rate'].values[0]:.1f}%")

    with col2:
        sku_df  = unified[unified["SKU"] == sel_sku].sort_values("Date")
        sku_hist = sku_df[sku_df["Units_Sold"].notna()]
        sku_fut  = sku_df[sku_df["Units_Sold"].isna()]

        fig = go.Figure()
        # CI band
        fig.add_trace(go.Scatter(
            x=pd.concat([sku_hist["Date"], sku_hist["Date"][::-1]]),
            y=pd.concat([sku_hist["upper_80"], sku_hist["lower_80"][::-1]]),
            fill="toself", fillcolor="rgba(255,152,0,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% CI (historical)"))
        # Actual
        fig.add_trace(go.Scatter(
            x=sku_hist["Date"], y=sku_hist["Units_Sold"],
            mode="lines", name="Actual",
            line=dict(color="#2196F3", width=1.5)))
        # Forecast
        fig.add_trace(go.Scatter(
            x=sku_hist["Date"], y=sku_hist["forecast"],
            mode="lines", name="Forecast",
            line=dict(color="#FF9800", width=2)))
        # Future
        if not sku_fut.empty:
            fig.add_trace(go.Scatter(
                x=sku_fut["Date"], y=sku_fut["forecast"],
                mode="lines", name="Future (28d)",
                line=dict(color="#4CAF50", width=2, dash="dash")))
            fig.add_trace(go.Scatter(
                x=pd.concat([sku_fut["Date"], sku_fut["Date"][::-1]]),
                y=pd.concat([sku_fut["upper_80"], sku_fut["lower_80"][::-1]]),
                fill="toself", fillcolor="rgba(76,175,80,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="80% CI (future)"))
        # Cutoff line
        fig.add_vline(x=cutoff, line_dash="dot", line_color="gray",
                      annotation_text="forecast →", annotation_position="top right")
        fig.update_layout(
            template="plotly_dark", height=400,
            title=f"SKU: {sel_sku}",
            margin=dict(l=0,r=0,t=40,b=0),
            xaxis_title="Date", yaxis_title="Units Sold",
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    # ── Error distribution for this SKU ───────────────────────────────────────
    if not sku_hist.empty:
        errors = sku_hist["forecast"] - sku_hist["Units_Sold"]
        fig_err = px.histogram(errors, nbins=40,
                               title="Forecast Error Distribution (Forecast − Actual)",
                               color_discrete_sequence=["#FF9800"],
                               template="plotly_dark")
        fig_err.add_vline(x=0, line_dash="dash", line_color="tomato")
        fig_err.update_layout(height=250, margin=dict(l=0,r=0,t=40,b=0),
                              xaxis_title="Error (units)", yaxis_title="Count",
                              showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 LightGBM vs Prophet — Model Comparison")
    st.markdown("---")

    lgbm_e  = eval_df[eval_df["model"] == "lgbm"]
    prop_e  = eval_df[eval_df["model"] == "prophet"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LGBM Median MAE",  f"{lgbm_e['MAE'].median():.2f}")
    c2.metric("LGBM Median MAPE", f"{lgbm_e['MAPE'].median():.1f}%")
    c3.metric("Prophet Median MAE",  f"{prop_e['MAE'].median():.2f}")
    c4.metric("Prophet CI Coverage", f"{prop_e['CI_Coverage_80'].mean():.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # MAE distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=lgbm_e["MAE"],   name="LightGBM",
                                   marker_color="#2196F3", opacity=0.8, nbinsx=20))
        fig.add_trace(go.Histogram(x=prop_e["MAE"],   name="Prophet",
                                   marker_color="#FF9800", opacity=0.7, nbinsx=20))
        fig.update_layout(template="plotly_dark", barmode="overlay", height=300,
                          title="MAE Distribution per SKU",
                          xaxis_title="MAE (units)", yaxis_title="# SKUs",
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # CI Coverage distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=lgbm_e["CI_Coverage_80"], name="LightGBM",
                                    marker_color="#2196F3", opacity=0.8, nbinsx=20))
        fig2.add_trace(go.Histogram(x=prop_e["CI_Coverage_80"], name="Prophet",
                                    marker_color="#FF9800", opacity=0.7, nbinsx=20))
        fig2.add_vline(x=80, line_dash="dash", line_color="tomato",
                       annotation_text="Target 80%")
        fig2.update_layout(template="plotly_dark", barmode="overlay", height=300,
                           title="80% CI Coverage Distribution",
                           xaxis_title="Coverage (%)", yaxis_title="# SKUs",
                           margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ── SKU-level comparison table ─────────────────────────────────────────────
    st.subheader("📋 Per-SKU Evaluation Table")
    sort_by = st.selectbox("Sort by", ["MAE","MAPE","CI_Coverage_80","Stockout_Rate"])
    asc     = st.checkbox("Ascending", value=True)
    show_df = eval_df.sort_values(sort_by, ascending=asc).reset_index(drop=True)
    show_df["model"] = show_df["model"].map({"lgbm":"🔵 LightGBM","prophet":"🟠 Prophet"})
    st.dataframe(show_df.style.background_gradient(subset=["MAE","MAPE"], cmap="RdYlGn_r")
                              .background_gradient(subset=["CI_Coverage_80"], cmap="RdYlGn"),
                 use_container_width=True, height=400)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — 28-DAY FORECAST
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗓️ 28-Day Forecast":
    st.title("🗓️ 28-Day Demand Forecast")
    st.caption(f"Forward-looking forecasts from {(cutoff + pd.Timedelta(days=1)).date()} onwards")
    st.markdown("---")

    fut_df = filter_skus(future)

    if fut_df.empty:
        st.warning("No future forecast rows found. Re-run the Prophet pipeline to generate 28-day ahead forecasts.")
    else:
        # ── Heatmap ───────────────────────────────────────────────────────────
        st.subheader("🌡️ Forecast Heatmap — All SKUs × Next 28 Days")
        pivot = fut_df.pivot_table(index="SKU", columns="Date",
                                   values="forecast", aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                        labels=dict(color="Forecast Units"))
        fig.update_layout(template="plotly_dark", height=500,
                          margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_title="Date", yaxis_title="SKU")
        st.plotly_chart(fig, use_container_width=True)

        # ── Top 10 highest demand SKUs ─────────────────────────────────────────
        st.subheader("🏆 Top 10 Highest Forecast Demand (Next 28 Days)")
        top10 = (fut_df.groupby("SKU")["forecast"].sum()
                       .sort_values(ascending=False).head(10).reset_index())
        top10.columns = ["SKU","Total Forecast (28d)"]
        fig2 = px.bar(top10, x="Total Forecast (28d)", y="SKU", orientation="h",
                      color="Total Forecast (28d)", color_continuous_scale="YlOrRd",
                      template="plotly_dark")
        fig2.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
                           showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

        # ── Raw table ──────────────────────────────────────────────────────────
        with st.expander("📄 View raw forecast table"):
            display = fut_df[["SKU","Date","model_used","forecast","lower_80","upper_80"]].copy()
            display["Date"] = display["Date"].dt.date
            display = display.round(2)
            st.dataframe(display, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — STOCKOUT RISK
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Stockout Risk":
    st.title("⚠️ Stockout Risk Monitor")
    st.caption("SKUs where forecast falls below actual demand — inventory replenishment signals")
    st.markdown("---")

    # Stockout = forecast < actual (under-forecast = risk of running out)
    risk_df = hist_f.copy()
    risk_df["under_forecast"] = risk_df["forecast"] < risk_df["Units_Sold"]
    risk_df["gap"] = (risk_df["Units_Sold"] - risk_df["forecast"]).clip(lower=0)

    sku_risk = (risk_df.groupby(["SKU","model_used"])
                .agg(stockout_rate=("under_forecast","mean"),
                     avg_gap=("gap","mean"),
                     total_gap=("gap","sum"))
                .reset_index())
    sku_risk["stockout_pct"] = (sku_risk["stockout_rate"] * 100).round(1)
    sku_risk = sku_risk.sort_values("stockout_pct", ascending=False)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("High-Risk SKUs (>55%)",  str((sku_risk["stockout_pct"] > 55).sum()))
    c2.metric("Avg Stockout Rate",      f"{sku_risk['stockout_pct'].mean():.1f}%")
    c3.metric("Avg Demand Gap",         f"{sku_risk['avg_gap'].mean():.1f} units/day")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Top 15 Highest-Risk SKUs")
        top15 = sku_risk.head(15)
        fig = px.bar(top15, x="stockout_pct", y="SKU", orientation="h",
                     color="stockout_pct",
                     color_continuous_scale=["#4CAF50","#FFC107","#F44336"],
                     template="plotly_dark",
                     labels={"stockout_pct":"Stockout Rate (%)"})
        fig.add_vline(x=50, line_dash="dash", line_color="white", opacity=0.4,
                      annotation_text="50%")
        fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0),
                          showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📦 Avg Daily Demand Gap (units under-forecasted)")
        fig2 = px.scatter(sku_risk, x="stockout_pct", y="avg_gap",
                          color="model_used", hover_data=["SKU"],
                          color_discrete_map={"lgbm":"#2196F3","prophet":"#FF9800"},
                          template="plotly_dark",
                          labels={"stockout_pct":"Stockout Rate (%)","avg_gap":"Avg Gap (units)"})
        fig2.add_hline(y=sku_risk["avg_gap"].mean(), line_dash="dot",
                       line_color="gray", annotation_text="avg gap")
        fig2.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Risk table ─────────────────────────────────────────────────────────────
    st.subheader("📋 Full Stockout Risk Table")
    display = sku_risk[["SKU","model_used","stockout_pct","avg_gap","total_gap"]].copy()
    display.columns = ["SKU","Model","Stockout Rate (%)","Avg Gap (units)","Total Gap (units)"]
    display["Model"] = display["Model"].map({"lgbm":"🔵 LightGBM","prophet":"🟠 Prophet"})
    display = display.round(2)
    st.dataframe(display.style.background_gradient(subset=["Stockout Rate (%)"],
                 cmap="RdYlGn_r"), use_container_width=True, height=400)
