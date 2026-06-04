# Airflow Pipeline Setup

## Quickstart (local)

```bash
# 1. Install Airflow
pip install apache-airflow==2.9.1

# 2. Set project root env var
export FORECAST_PROJECT_ROOT=/path/to/Demand-forecasting-engine

# 3. Point Airflow to this DAGs folder
export AIRFLOW__CORE__DAGS_FOLDER=$FORECAST_PROJECT_ROOT/airflow/dags

# 4. Initialise Airflow DB
airflow db init

# 5. Start scheduler + webserver
airflow scheduler &
airflow webserver --port 8080

# 6. Open http://localhost:8080 → find "demand_forecast_pipeline" → trigger ▶
```

## DAG Structure

```
start
  └── ingest_data          Load & validate raw CSV → cleaned.parquet
        └── build_features  Lag/rolling/calendar features → features.parquet
              ├── train_lgbm     LightGBM CV → lgbm_forecasts.parquet
              └── run_prophet    Prophet 28d → prophet_forecasts.parquet
                    └── merge_forecasts  Unified table → unified_forecasts.parquet
                          └── evaluate_forecasts  MAE / MAPE / CI per SKU
                                └── pipeline_summary  Print run report
                                      └── end
```

## Schedule
Runs every **Monday at 06:00** — mirrors Zara's twice-weekly store replenishment cycle.

```
0 6 * * 1
```

## Key outputs per run

| File | Description |
|---|---|
| `data/processed/cleaned.parquet` | Validated raw data |
| `data/processed/features.parquet` | Engineered feature matrix |
| `outputs/lgbm_forecasts.parquet` | LightGBM point forecasts + CI |
| `outputs/prophet_forecasts.parquet` | Prophet forecasts + CI (28d future) |
| `outputs/unified_forecasts.parquet` | Merged forecast table (all 100 SKUs) |
| `outputs/unified_evaluation.csv` | MAE, MAPE, CI coverage per SKU |

## XCom metrics pushed each run

| Task | Key | Value |
|---|---|---|
| ingest_data | n_rows, n_skus | Row count, SKU count |
| build_features | feature_shape | (rows, cols) |
| train_lgbm | lgbm_skus | # SKUs trained |
| run_prophet | prophet_skus | # SKUs trained |
| merge_forecasts | unified_rows, future_rows | Total + future rows |
| evaluate_forecasts | lgbm_median_mae, lgbm_median_mape, prop_ci_coverage | Key metrics |
