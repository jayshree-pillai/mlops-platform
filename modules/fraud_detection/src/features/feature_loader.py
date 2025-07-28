import pandas as pd
from pathlib import Path
from .feature_processor import FeatureProcessor

def load_processor(path):
    return FeatureProcessor.load(path)

def load_row_by_id(tx_id, parquet_path="s3://mlops-fraud-dev/data/processed/test.parquet"):
    df = pd.read_parquet(parquet_path)
    row = df[df["transactionID"] == tx_id]
    if row.empty:
        raise ValueError(f"Transaction ID {tx_id} not found.")
    return row

def transform_row(row: pd.DataFrame, processor: FeatureProcessor):
    return processor.transform(row)