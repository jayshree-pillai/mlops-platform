import boto3
import os

BUCKET = "complaint-classifier-jp2025"
PREFIX = "rag/semantic_index/"
LOCAL_DIR = "semantic_index"

os.makedirs(LOCAL_DIR, exist_ok=True)
s3 = boto3.client("s3")

objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX).get("Contents", [])
for obj in objects:
    key = obj["Key"]
    filename = key.split("/")[-1]
    local_path = os.path.join(LOCAL_DIR, filename)
    s3.download_file(BUCKET, key, local_path)
    print(f"✅ Downloaded {filename} to {local_path}")
