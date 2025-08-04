import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
import uuid
from datetime import datetime
from src.features.feature_processor import FeatureProcessor

def log_features_to_store(X, y, bucket, s3_prefix, processor=None, model_id=None):
    if model_id is None:
        model_id = str(uuid.uuid4())

    try:
        if processor and hasattr(processor, "feature_columns"):
            df = pd.DataFrame(X, columns=processor.feature_columns)
        else:
            raise ValueError("Processor missing or invalid.")
    except Exception as e:
        print(f"⚠️ Processor not found or failed: {e} — using default column names.")
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    df["model_id"] = model_id
    df["timestamp"] = datetime.utcnow().isoformat()
    df["schema_hash"] = getattr(processor, "schema_hash", "unknown")

    table = pa.Table.from_pandas(df)
    local_path = "/tmp/features.parquet"
    pq.write_table(table, local_path)

    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, f"{s3_prefix}/features_{model_id}.parquet")

    print(f"✅ Uploaded features to s3://{bucket}/{s3_prefix}/features_{model_id}.parquet")
    return model_id
