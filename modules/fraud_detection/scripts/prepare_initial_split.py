import pandas as pd
import boto3
import io
from sklearn.model_selection import train_test_split

# Config
BUCKET = "mlops-fraud-dev"
RAW_KEY = "data/raw/paysim_full.csv"
PROCESSED_PREFIX = "data/processed/"

s3 = boto3.client("s3")

# Load CSV from S3
obj = s3.get_object(Bucket=BUCKET, Key=RAW_KEY)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

# 50/50 split
offline_df, drift_df = train_test_split(df, test_size=0.5, random_state=42, stratify=df['isFraud'])

# Train/val/test from offline_df
train_df, temp_df = train_test_split(offline_df, test_size=0.4, random_state=42, stratify=offline_df['isFraud'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['isFraud'])

# Save to S3 as parquet
def save_parquet_to_s3(df, key):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET, key)

save_parquet_to_s3(train_df, f"{PROCESSED_PREFIX}train.parquet")
save_parquet_to_s3(val_df, f"{PROCESSED_PREFIX}val.parquet")
save_parquet_to_s3(test_df, f"{PROCESSED_PREFIX}test.parquet")
save_parquet_to_s3(drift_df, f"{PROCESSED_PREFIX}drift_stream.parquet")

print("All splits saved to S3.")
