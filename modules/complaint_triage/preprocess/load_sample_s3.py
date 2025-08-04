import pandas as pd
import boto3
import io

BUCKET = "complaint-classifier-jp2025"
KEY = "data/complaints_processed.csv"
TEXT_COLUMN = "narrative"
N_SAMPLES = 1000
OUTPUT_PATH = "../data/complaints_sample.csv"


def extract_sample_from_s3():
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"❌ Column '{TEXT_COLUMN}' not found.")

    df = df[[TEXT_COLUMN]].dropna().sample(n=min(N_SAMPLES, len(df)), random_state=42)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Sample saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_sample_from_s3()
