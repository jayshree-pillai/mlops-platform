import pandas as pd
import numpy as np
import os
import io
import boto3
import s3fs

from pathlib import Path
from feature_processor import FeatureProcessor

# S3 target
BUCKET = "mlops-fraud-dev"
PREFIX = "data/processed_np/"
s3 = boto3.client('s3')

def upload_array(arr, s3_key):
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET, f"{PREFIX}{s3_key}")

# Load parquet
LOCAL_PARQUET = "data/processed"
splits = {
    split: pd.read_parquet(f"s3://mlops-fraud-dev/data/processed/{split}.parquet")
    for split in ["train", "val", "test"]
}

# Fit + upload train
fp = FeatureProcessor()
X_train = fp.fit_transform(splits["train"])
y_train = splits["train"]["isFraud"].values
upload_array(X_train, "train_X.npy")
upload_array(y_train, "train_y.npy")

# Transform + upload val/test
for split in ["val", "test"]:
    X = fp.transform(splits[split])
    y = splits[split]["isFraud"].values
    upload_array(X, f"{split}_X.npy")
    upload_array(y, f"{split}_y.npy")

# Save transformer
Path("artifacts").mkdir(exist_ok=True)
fp.save("artifacts/feature_processor.pkl")

print("All features processed + .npy arrays uploaded to S3.")
