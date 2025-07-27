import boto3
import io
import numpy as np

BUCKET = "mlops-fraud-dev"
PREFIX = "data/processed_np/"

def load_npy_from_s3(key):
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    s3.download_fileobj(BUCKET, f"{PREFIX}{key}", buffer)
    buffer.seek(0)
    return np.load(buffer)
