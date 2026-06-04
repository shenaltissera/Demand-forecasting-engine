"""
Zara-Inspired Demand Forecasting Pipeline
==========================================
Runs every Monday at 06:00 — mirrors Zara's twice-weekly replenishment cycle.

DAG structure:
                         ┌─── train_lgbm ───┐
  ingest ── features ────┤                  ├── merge ── evaluate ── notify
                         └─── run_prophet ──┘

Tasks:
  t1  ingest_data      Load & validate raw CSV, save cleaned parquet
  t2  build_features   Lag, rolling, calendar, categorical features
  t3  train_lgbm       LightGBM with TimeSeriesSplit CV (stable SKUs)
  t4  run_prophet      Prophet for volatile/intermittent SKUs
  t5  merge_forecasts  Unified forecast table (LightGBM + Prophet)
  t6  evaluate         MAPE, MAE, CI coverage per SKU
  t7  pipeline_summary Print final report + push metrics to XCom
"""

import os, sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# ── Project root on the Airflow worker ───────────────────────────────────────
PROJECT_ROOT = os.environ.get("FORECAST_PROJECT_ROOT",
                              os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, PROJECT_ROOT)

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner":            "zara-forecasting",
    "retries":          2,
    "retry_delay":      timedelta(minutes=10),
    "execution_timeout": timedelta(hours=2),
    "email_on_failure": False,
}

# ── Task callables ────────────────────────────────────────────────────────────

def task_ingest(**context):
    """Load raw CSV, validate, save cleaned parquet."""
    os.chdir(PROJECT_ROOT)
    from src.ingestion.load_data import load_raw_data, validate, save_processed
    df = load_raw_data()
    df = validate(df)
    save_processed(df)
    context["ti"].xcom_push(key="n_rows",  value=len(df))
    context["ti"].xcom_push(key="n_skus",  value=int(df["SKU"].nunique()))
    print(f"✓ Ingested {len(df):,} rows | {df['SKU'].nunique()} SKUs")


def task_build_features(**context):
    """Feature engineering: lag, rolling, calendar, categorical encoding."""
    import pandas as pd
    os.chdir(PROJECT_ROOT)
    from src.features.feature_engineering import build_features
    df = pd.read_parquet("data/processed/cleaned.parquet")
    df = build_features(df)
    df.to_parquet("data/processed/features.parquet", index=False)
    context["ti"].xcom_push(key="feature_shape", value=str(df.shape))
    print(f"✓ Feature matrix: {df.shape}")


def task_train_lgbm(**context):
    """Train LightGBM with 5-fold TimeSeriesSplit CV."""
    import pandas as pd
    os.chdir(PROJECT_ROOT)
    from src.models.lgbm_model import train, predict
    df = pd.read_parquet("data/processed/features.parquet")
    model, available = train(df)

    # Generate & save forecasts immediately after training
    forecasts = predict(df)
    forecasts.to_parquet("outputs/lgbm_forecasts.parquet", index=False)
    context["ti"].xcom_push(key="lgbm_skus", value=int(forecasts["SKU"].nunique()))
    print(f"✓ LightGBM trained | {forecasts['SKU'].nunique()} SKUs forecast")


def task_run_prophet(**context):
    """Train Prophet on volatile/intermittent SKUs, generate 28-day forecast."""
    import pandas as pd
    os.chdir(PROJECT_ROOT)
    from src.models.prophet_model import run_prophet_pipeline
    df = pd.read_parquet("data/processed/cleaned.parquet")
    forecasts = run_prophet_pipeline(df, periods=28, save_models=True)
    n_skus = int(forecasts["SKU"].nunique()) if not forecasts.empty else 0
    context["ti"].xcom_push(key="prophet_skus", value=n_skus)
    print(f"✓ Prophet trained | {n_skus} SKUs forecast")


def task_merge_forecasts(**context):
    """Merge LightGBM + Prophet into unified_forecasts.parquet."""
    import pandas as pd
    os.chdir(PROJECT_ROOT)
    from src.models.ensemble import load_all, build_unified
    actuals, lgbm_fc, prop_fc, routing = load_all()
    unified = build_unified(actuals, lgbm_fc, prop_fc, routing)
    unified.to_parquet("outputs/unified_forecasts.parquet", index=False)
    context["ti"].xcom_push(key="unified_rows", value=len(unified))
    context["ti"].xcom_push(key="future_rows",
                            value=int(unified["Units_Sold"].isna().sum()))
    print(f"✓ Unified table: {len(unified):,} rows | "
          f"future rows: {unified['Units_Sold'].isna().sum():,}")


def task_evaluate(**context):
    """Compute MAPE, MAE, CI coverage, stockout rate per SKU."""
    import pandas as pd, numpy as np
    os.chdir(PROJECT_ROOT)
    from src.models.ensemble import load_all, build_unified, evaluate

    actuals, lgbm_fc, prop_fc, routing = load_all()
    unified  = build_unified(actuals, lgbm_fc, prop_fc, routing)
    report   = evaluate(unified)
    report.to_csv("outputs/unified_evaluation.csv", index=False)

    # Push summary metrics to XCom
    lgbm_r   = report[report["model"] == "lgbm"]
    prop_r   = report[report["model"] == "prophet"]
    context["ti"].xcom_push(key="lgbm_median_mae",  value=round(float(lgbm_r["MAE"].median()), 2))
    context["ti"].xcom_push(key="lgbm_median_mape", value=round(float(lgbm_r["MAPE"].median()), 2))
    context["ti"].xcom_push(key="prop_ci_coverage", value=round(float(prop_r["CI_Coverage_80"].mean()), 2))
    print("✓ Evaluation complete")


def task_pipeline_summary(**context):
    """Pull all XCom metrics and print a clean run summary."""
    ti = context["ti"]

    def pull(task_id, key, default="N/A"):
        val = ti.xcom_pull(task_ids=task_id, key=key)
        return val if val is not None else default

    n_rows          = pull("ingest_data",       "n_rows")
    n_skus          = pull("ingest_data",       "n_skus")
    feat_shape      = pull("build_features",    "feature_shape")
    lgbm_skus       = pull("train_lgbm",        "lgbm_skus")
    prophet_skus    = pull("run_prophet",       "prophet_skus")
    unified_rows    = pull("merge_forecasts",   "unified_rows")
    future_rows     = pull("merge_forecasts",   "future_rows")
    lgbm_mae        = pull("evaluate_forecasts","lgbm_median_mae")
    lgbm_mape       = pull("evaluate_forecasts","lgbm_median_mape")
    prop_cov        = pull("evaluate_forecasts","prop_ci_coverage")

    run_date = context["ds"]

    summary = f"""
╔══════════════════════════════════════════════════════╗
║       DEMAND FORECAST PIPELINE — RUN SUMMARY        ║
║       Run date : {run_date}                          ║
╠══════════════════════════════════════════════════════╣
║  DATA                                                ║
║    Rows ingested      : {str(n_rows):>10}            ║
║    Total SKUs         : {str(n_skus):>10}            ║
║    Feature shape      : {str(feat_shape):>20}        ║
╠══════════════════════════════════════════════════════╣
║  MODELS                                              ║
║    LightGBM SKUs      : {str(lgbm_skus):>10}         ║
║    Prophet SKUs       : {str(prophet_skus):>10}       ║
║    Unified rows       : {str(unified_rows):>10}      ║
║    Future (28d) rows  : {str(future_rows):>10}       ║
╠══════════════════════════════════════════════════════╣
║  METRICS                                             ║
║    LightGBM Median MAE  : {str(lgbm_mae):>8} units  ║
║    LightGBM Median MAPE : {str(lgbm_mape):>7}%       ║
║    Prophet CI Coverage  : {str(prop_cov):>7}%        ║
╚══════════════════════════════════════════════════════╝
"""
    print(summary)
    context["ti"].xcom_push(key="pipeline_summary", value=summary)


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="demand_forecast_pipeline",
    default_args=default_args,
    description="Zara-inspired weekly SKU demand forecasting pipeline",
    schedule_interval="0 6 * * 1",   # Every Monday 06:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,               # Never run two pipelines in parallel
    tags=["forecasting", "inventory", "zara"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    t1 = PythonOperator(task_id="ingest_data",       python_callable=task_ingest,           provide_context=True)
    t2 = PythonOperator(task_id="build_features",    python_callable=task_build_features,   provide_context=True)
    t3 = PythonOperator(task_id="train_lgbm",        python_callable=task_train_lgbm,       provide_context=True)
    t4 = PythonOperator(task_id="run_prophet",       python_callable=task_run_prophet,      provide_context=True)
    t5 = PythonOperator(task_id="merge_forecasts",   python_callable=task_merge_forecasts,  provide_context=True)
    t6 = PythonOperator(task_id="evaluate_forecasts",python_callable=task_evaluate,         provide_context=True)
    t7 = PythonOperator(task_id="pipeline_summary",  python_callable=task_pipeline_summary, provide_context=True)

    # ── Wiring ─────────────────────────────────────────────────────────────────
    #
    #                    ┌── t3: train_lgbm ──┐
    # start ─ t1 ─ t2 ──┤                    ├── t5 ─ t6 ─ t7 ─ end
    #                    └── t4: run_prophet ─┘
    #
    start >> t1 >> t2 >> [t3, t4] >> t5 >> t6 >> t7 >> end
