import pandas as pd
import os

RAW_DATA_PATH = "data/raw/retail_inventory.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned.parquet"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download from: https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset\n"
            "and place the CSV in data/raw/"
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {len(df):,} rows | {df['SKU'].nunique():,} SKUs")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation and type enforcement."""
    required_cols = ["SKU", "Date", "Units_Sold", "Price", "Inventory_Level"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["SKU", "Date", "Units_Sold"])
    df["Units_Sold"] = df["Units_Sold"].clip(lower=0)  # no negative demand
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH):
    df.to_parquet(path, index=False)
    print(f"Saved processed data to {path}")


if __name__ == "__main__":
    df = load_raw_data()
    df = validate(df)
    save_processed(df)
