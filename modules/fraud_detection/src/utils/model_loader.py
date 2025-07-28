import joblib
import boto3
import os

BUCKET = "mlops-fraud-dev"

def get_model_and_processor(version="v1"):
    model_path = f"/tmp/model_{version}.pkl"
    processor_path = f"/tmp/processor_{version}.pkl"

    s3 = boto3.client("s3")
    s3.download_file(BUCKET, f"models/{version}/model.pkl", model_path)
    s3.download_file(BUCKET, f"models/{version}/feature_processor.pkl", processor_path)

    model = joblib.load(model_path)
    processor = joblib.load(processor_path)
    return model, processor
