FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY modules/complaint_triage/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ⬇️ Only copy required subfolders and files
COPY modules/complaint_triage/app /app/app
COPY modules/complaint_triage/faiss_index /app/faiss_index

# Optional: if you need these during runtime (e.g., local test endpoints)
# COPY modules/complaint_triage/data /app/data
# COPY modules/complaint_triage/prompt /app/prompt

ENV PORT=8000

ENTRYPOINT ["bash", "-c", "python app/download_index_from_s3.py && uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
