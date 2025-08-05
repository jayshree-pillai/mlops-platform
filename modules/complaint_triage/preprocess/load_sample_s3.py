import pandas as pd
import boto3
import io

BUCKET = "complaint-classifier-jp2025"
KEY = "data/complaints_processed.csv"
SAMPLE_KEY = "data/complaints_sample.csv"
TEXT_COLUMN = "narrative"
N_SAMPLES = 1000

def extract_sample_from_s3():
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()), header=None)
    df.columns = ["complaint_id", "product", "narrative"]  # manually assign

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"❌ Column '{TEXT_COLUMN}' not found.")
    df = df[["complaint_id", TEXT_COLUMN]].dropna()
    df = df.sample(n=min(N_SAMPLES, len(df)), random_state=42)

    # Save locally first
    df.to_csv("complaints_sample.csv", index=False)

    # Upload back to S3
    s3.upload_file("complaints_sample.csv", BUCKET, SAMPLE_KEY)
    print(f"✅ Uploaded sample to: s3://{BUCKET}/{SAMPLE_KEY}")

if __name__ == "__main__":
    extract_sample_from_s3()
