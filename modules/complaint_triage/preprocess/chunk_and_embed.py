import pandas as pd
import os
import boto3
import re
from bs4 import BeautifulSoup
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.preprocessing import normalize
import faiss
# === Config
INPUT_CSV = "complaints_sample.csv"
OUT_DIR = "semantic_index"
BUCKET = "complaint-classifier-jp2025"
PREFIX = "rag/semantic_index/"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 100

# === Clean-up Function
def clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()  # strip HTML
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(Regards,|Sincerely,|Forwarded message:)", "", text, flags=re.I)
    return text

# === Load and clean complaints
df = pd.read_csv(INPUT_CSV, dtype={"complaint_id": str})
df = df.dropna(subset=["narrative", "complaint_id"])

texts = []
ids = []

for _, row in df.iterrows():
    texts.append(clean_text(row["narrative"]))
    ids.append(str(row["complaint_id"]))

# === Token-based chunking
splitter = TokenTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    encoding_name="cl100k_base"
)

docs = splitter.create_documents(
    texts,
    metadatas=[{"complaint_id": cid} for cid in ids]
)

# === Embed and create FAISS index use Cosine Norm
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents=docs, embedding=embedding)
db.save_local(OUT_DIR)

print(f"✅ FAISS index created with {len(docs)} chunks")
print(f"✅ Saved index to {OUT_DIR}")

# === Upload to S3
s3 = boto3.client("s3")

for file in os.listdir(OUT_DIR):
    path = os.path.join(OUT_DIR, file)
    s3.upload_file(path, BUCKET, PREFIX + file)
    print(f"✅ Uploaded {file} to s3://{BUCKET}/{PREFIX}{file}")
