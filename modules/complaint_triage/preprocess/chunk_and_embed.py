import pandas as pd
import os
import boto3
import re
import pickle
import numpy as np
from bs4 import BeautifulSoup
from langchain.text_splitter import TokenTextSplitter
from openai import OpenAI
from sklearn.preprocessing import normalize
import faiss
from tqdm import tqdm
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import mlflow
import json

# === Config
INPUT_CSV = "complaints_sample.csv"
OUT_DIR = "semantic_index"
BUCKET = "complaint-classifier-jp2025"
PREFIX = "rag/semantic_index/"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 100
PCA_DIM = 256  # try 384 if needed
EMBED_MODEL = "text-embedding-3-large"


# MLflow config (no change to where YOU upload artifacts)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "complaint-embeddings"))

# Version stamp reused for run name/tags
VERSION = os.environ.get(
    "FAISS_VERSION",
    f"faiss-pca-v2.0-{datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')}"
)

# === Init OpenAI
client = OpenAI()
def batch_embed(texts, batch_size=20, retry_delay=5):
    client = OpenAI()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="🔢 Embedding in batches"):
        batch = texts[i:i+batch_size]
        try:
            res = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch
            )
            batch_embeds = [np.array(e.embedding, dtype=np.float32) for e in res.data]
            all_embeddings.extend(batch_embeds)
        except Exception as e:
            print(f"❌ Error on batch {i}–{i+batch_size}: {e}")
            print(f"⏳ Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            continue

    return np.array(all_embeddings)

def get_embedding(text):
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return np.array(res.data[0].embedding, dtype=np.float32)

# === Clean-up Function
def clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(Regards,|Sincerely,|Forwarded message:)", "", text, flags=re.I)
    return text

# === Load and chunk
df = pd.read_csv(INPUT_CSV, dtype={"complaint_id": str}).dropna(subset=["narrative", "complaint_id"])
texts = [clean_text(t) for t in df["narrative"].tolist()]
ids = df["complaint_id"].astype(str).tolist()

# === Token-based chunking
splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, encoding_name="cl100k_base")
docs = splitter.create_documents(texts, metadatas=[{"complaint_id": cid} for cid in ids])

# === Embed
raw_texts = [doc.page_content for doc in docs]
metadata = [doc.metadata for doc in docs]

# === Build + log run
start = time.time()
with mlflow.start_run(run_name=VERSION):
    # Params/tags that describe this build
    mlflow.set_tags({
        "component": "chunk_and_embed",
        "stage": "build-faiss",
        "version": VERSION,
    })
    # === Run embedding
    print(f"🔢 Embedding {len(raw_texts)} chunks using OpenAI batch...")
    embeddings = batch_embed(raw_texts, batch_size=20)
    embed_dim = embeddings.shape[1]
    mlflow.log_params({
        "embedding_model": EMBED_MODEL,
        "pca_k": int(PCA_DIM),
        "embed_dim": int(embed_dim),
        "n_docs": int(len(raw_texts)),
        "chunk_size": int(CHUNK_SIZE),
        "chunk_overlap": int(CHUNK_OVERLAP),
    })
    embeddings = normalize(embeddings, axis=1)

    # === PCA
    print(f"🧪 Applying PCA → {PCA_DIM} dims...")
    pca = faiss.PCAMatrix(embeddings.shape[1], PCA_DIM)
    pca.train(embeddings)
    embeddings_pca = pca.apply_py(embeddings)
    embeddings_pca = normalize(embeddings_pca, axis=1)  # Re-normalize for cosine

    # === FAISS Index (cosine similarity via inner product on unit vectors)
    index = faiss.IndexFlatIP(PCA_DIM)
    index.add(embeddings_pca)

    # === Save all artifacts
    os.makedirs(OUT_DIR, exist_ok=True)
    local_index = f"{OUT_DIR}/index.faiss"
    local_pca = f"{OUT_DIR}/pca.transform"
    local_meta = f"{OUT_DIR}/index.pkl"

    faiss.write_index(index, local_index)
    faiss.write_VectorTransform(pca, local_pca)

    with open(local_meta, "wb") as f:
        pickle.dump({"texts": raw_texts, "metadata": metadata}, f)

    print(f"✅ FAISS index + PCA saved to `{OUT_DIR}`")


    # === Upload to S3
    s3 = boto3.client("s3")

    version_prefix = f"{PREFIX}versions/{VERSION}/"
    current_prefix = f"{PREFIX}current/"
    files = [
        ("index.faiss", local_index),
        ("pca.transform", local_pca),
        ("index.pkl", local_meta),
    ]
    # Upload versioned files
    for name, path in files:
        s3.upload_file(path, BUCKET, version_prefix + name)
        print(f"☁️ Uploaded `{name}` to s3://{BUCKET}/{version_prefix}{name}")

    # Create manifest and upload it as well
    manifest = {
        "version": VERSION,
        "s3_prefix": f"s3://{BUCKET}/{version_prefix}",
        "files": [n for n, _ in files] + ["manifest.json"],
        "n_docs": int(len(raw_texts)),
        "pca_k": int(PCA_DIM),
        "embed_dim": int(embed_dim),
        "built_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
    }
    local_manifest = os.path.join(OUT_DIR, "manifest.json")
    Path(local_manifest).write_text(json.dumps(manifest, indent=2))
    s3.upload_file(local_manifest, BUCKET, version_prefix + "manifest.json")
    print(f"☁️ Uploaded `manifest.json` to s3://{BUCKET}/{version_prefix}manifest.json")

    # Mirror to 'current/' (acts like a pointer for consumers)
    for name, _ in files + [("manifest.json", local_manifest)]:
        s3.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": BUCKET, "Key": version_prefix + name},
            Key=current_prefix + name,
        )
    # Also write the current version marker
    s3.put_object(Bucket=BUCKET, Key=current_prefix + "version.txt", Body=VERSION.encode("utf-8"))
    print(f"🔗 Updated current/ to point at version {VERSION}")

    # Log minimal artifact(s) to MLflow (manifest only, since big files live in S3)
    mlflow.log_artifact(local_manifest, artifact_path="faiss")

    # Timing metric
    mlflow.log_metric("build_seconds", time.time() - start)

print("🏁 Done.")