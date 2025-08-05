import pandas as pd
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import boto3
import os


# === Load complaints
df = pd.read_csv("complaints_sample.csv")
texts = df["narrative"].dropna().tolist()

# === Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents(texts)

# === Embed
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)

out_dir = "semantic_index"
db.save_local(out_dir)

s3 = boto3.client("s3")
bucket = "complaint-classifier-jp2025"
prefix = "rag/semantic_index/"

for file in os.listdir(out_dir):
    path = os.path.join(out_dir, file)
    s3.upload_file(path, bucket, prefix + file)
    print(f"✅ Uploaded {file} to s3://{bucket}/{prefix}{file}")