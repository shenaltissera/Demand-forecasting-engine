import pandas as pd
import os

RAW_DATA_PATH = "data/raw/retail_store_inventory.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned.parquet"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download from Kaggle: retail-store-inventory-forecasting-dataset\n"
            "and place the CSV in data/raw/"
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
    print(f"✓ Loaded {len(df):,} rows | {df['Product_ID'].nunique()} SKUs | {df['Store_ID'].nunique()} stores")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation and type enforcement."""
    required_cols = ["Product_ID", "Store_ID", "Date", "Units_Sold", "Inventory_Level"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["Product_ID", "Date", "Units_Sold"])
    df["Units_Sold"] = df["Units_Sold"].clip(lower=0)
    df = df.sort_values(["Product_ID", "Store_ID", "Date"]).reset_index(drop=True)

    # Create a unique SKU key = Product x Store
    df["SKU"] = df["Product_ID"] + "_" + df["Store_ID"]
    print(f"✓ Validated | {df['SKU'].nunique()} SKU×Store combinations | {df['Units_Sold'].isna().sum()} nulls dropped")
    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✓ Saved processed data → {path}")


if __name__ == "__main__":
    df = load_raw_data()
    df = validate(df)
    save_processed(df)
