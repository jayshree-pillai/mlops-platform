import joblib
import boto3
import os

BUCKET = "mlops-fraud-dev"

def get_model_and_processor(version="v1"):
    bucket = os.environ["MODEL_BUCKET"]
    model_key = os.environ["MODEL_KEY"]
    schema_key = os.environ["SCHEMA_KEY"]

    model_path = "/tmp/model.pkl"
    processor_path = "/tmp/feature_processor.pkl"

    s3 = boto3.client("s3")
    s3.download_file(bucket, model_key, model_path)
    s3.download_file(bucket, schema_key, processor_path)

    model = joblib.load(model_path)
    processor = joblib.load(processor_path)
    return model, processor