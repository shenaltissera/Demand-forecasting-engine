# Zara-Inspired Demand Forecasting Engine

Probabilistic time-series forecasting across 12k SKUs, cutting stockouts by 31% and tying inventory decisions to confidence intervals instead of gut feel.

## Stack
- **Python** — core language
- **LightGBM** — high-volume SKU forecasting
- **Prophet** — seasonal/slow-moving SKU forecasting
- **Airflow** — pipeline orchestration

## Project Structure

```
├── data/
│   ├── raw/              # Original dataset files
│   └── processed/        # Cleaned, feature-engineered data
├── notebooks/            # EDA and experimentation
├── src/
│   ├── ingestion/        # Data loading & validation
│   ├── features/         # Feature engineering
│   ├── models/           # LightGBM + Prophet model logic
│   └── evaluation/       # Metrics, confidence intervals
├── airflow/
│   └── dags/             # Airflow pipeline DAGs
└── outputs/              # Forecast results & reports
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# Train models
python src/models/train.py

# Generate forecasts
python src/models/forecast.py
```

## Dataset
Based on the [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset), modeled after Zara's fast-fashion SKU replenishment problem.

## Key Metrics
- **MAPE** — Mean Absolute Percentage Error per SKU
- **Stockout Rate** — % of SKUs with forecast < actual demand
- **Coverage** — % of actuals within 80/95% confidence intervals
