from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append("/opt/airflow/project")  # adjust to your project root

default_args = {
    "owner": "zara-forecasting",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="demand_forecast_pipeline",
    default_args=default_args,
    description="Weekly Zara-inspired SKU demand forecasting pipeline",
    schedule_interval="0 6 * * 1",  # every Monday at 6am (matches Zara's weekly replenishment)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["forecasting", "inventory"],
) as dag:

    def ingest():
        from src.ingestion.load_data import load_raw_data, validate, save_processed
        df = load_raw_data()
        df = validate(df)
        save_processed(df)

    def build_features():
        import pandas as pd
        from src.features.feature_engineering import build_features
        df = pd.read_parquet("data/processed/cleaned.parquet")
        df = build_features(df)
        df.to_parquet("data/processed/features.parquet", index=False)

    def train_lgbm():
        import pandas as pd
        from src.models.lgbm_model import train
        df = pd.read_parquet("data/processed/features.parquet")
        train(df)

    def run_prophet():
        import pandas as pd
        from src.models.prophet_model import run_all_skus
        df = pd.read_parquet("data/processed/cleaned.parquet")
        forecasts = run_all_skus(df, periods=28)
        forecasts.to_parquet("outputs/prophet_forecasts.parquet", index=False)

    def evaluate():
        import pandas as pd
        from src.evaluation.metrics import evaluate
        df = pd.read_parquet("outputs/lgbm_forecasts.parquet")
        report = evaluate(df)
        report.to_csv("outputs/evaluation_report.csv", index=False)

    t1 = PythonOperator(task_id="ingest_data",        python_callable=ingest)
    t2 = PythonOperator(task_id="build_features",     python_callable=build_features)
    t3 = PythonOperator(task_id="train_lgbm",         python_callable=train_lgbm)
    t4 = PythonOperator(task_id="run_prophet",        python_callable=run_prophet)
    t5 = PythonOperator(task_id="evaluate_forecasts", python_callable=evaluate)

    t1 >> t2 >> [t3, t4] >> t5
