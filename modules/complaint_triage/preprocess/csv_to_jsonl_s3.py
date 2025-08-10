# csv_to_jsonl_s3.py
import os, sys, csv, json, io, boto3

BUCKET    = os.environ.get("BUCKET", "complaint-classifier-jp2025")
INPUT_KEY = os.environ.get("SAMPLE_KEY", "data/complaints_sample.csv")
OUTPUT_KEY= os.environ.get("OUT_KEY", "data/complaints_sample.jsonl")

ID_COL    = os.environ.get("ID_COL", "complaint_id")
TEXT_COL  = os.environ.get("TEXT_COL", "narrative")  # becomes "query" in JSONL

s3 = boto3.client("s3")

# read CSV from S3
obj = s3.get_object(Bucket=BUCKET, Key=INPUT_KEY)
f = io.TextIOWrapper(obj["Body"], encoding="utf-8", newline="")
r = csv.DictReader(f)

missing = [c for c in (ID_COL, TEXT_COL) if c not in r.fieldnames]
if missing:
    sys.exit(f"Missing columns in CSV: {missing}. Found: {r.fieldnames}")

lines = []
for row in r:
    cid = str(row.get(ID_COL, "")).strip()
    txt = (row.get(TEXT_COL) or "").strip()
    if not cid or not txt:
        continue
    lines.append(json.dumps({"complaint_id": cid, "query": txt}, ensure_ascii=False) + "\n")

body = "".join(lines).encode("utf-8")
s3.put_object(Bucket=BUCKET, Key=OUTPUT_KEY, Body=body)
print(f"Wrote JSONL to s3://{BUCKET}/{OUTPUT_KEY} ({len(lines)} rows)")
