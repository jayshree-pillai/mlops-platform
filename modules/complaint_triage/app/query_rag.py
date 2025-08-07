import os
import pickle
import faiss
import numpy as np
from openai import OpenAI
from sklearn.preprocessing import normalize
import boto3

# === Config
BUCKET = "complaint-classifier-jp2025"
PREFIX = "rag/semantic_index/"
OUT_DIR = "semantic_index"
INDEX_PATH = f"{OUT_DIR}/index.faiss"
PCA_PATH = f"{OUT_DIR}/pca.transform"
PICKLE_PATH = f"{OUT_DIR}/index.pkl"

def download_artifacts():
    s3 = boto3.client("s3")
    os.makedirs(OUT_DIR, exist_ok=True)
    for file in ["index.faiss", "pca.transform", "index.pkl"]:
        s3.download_file(BUCKET, PREFIX + file, f"{OUT_DIR}/{file}")

download_artifacts()

# === Load
index = faiss.read_index(INDEX_PATH)
pca = faiss.read_VectorTransform(PCA_PATH)
with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)
texts = data["texts"]
metadatas = data["metadata"]

# === Query
client = OpenAI()
query = input("❓ Ask your question: ").strip()
res = client.embeddings.create(model="text-embedding-3-large", input=[query])
query_vec = np.array(res.data[0].embedding, dtype=np.float32).reshape(1, -1)  # already normed
query_pca = pca.apply_py(query_vec)
query_pca = normalize(query_pca, axis=1)  # must normalize after PCA

# === Search
k = 5
scores, indices = index.search(query_pca, k)
print(f"\n🔥 Max similarity score: {max(scores[0]):.4f}")

THRESHOLD = 0.5
filtered = [
    (texts[i], metadatas[i], scores[0][rank])
    for rank, i in enumerate(indices[0])
    if scores[0][rank] >= THRESHOLD
]

if not filtered:
    print("\n⚠️ No relevant context found (score below threshold). LLM will not be called.")
    print("🧠 LLM Response: I'm sorry, I couldn't find enough relevant information to answer that.")
    exit()

# === Show matches
print("\n🔍 Top Matches:")
for i, (text, meta, score) in enumerate(filtered):
    print(f"\n[{i+1}] complaint_id: {meta['complaint_id']} | Score: {score:.4f}")
    print(text[:500])

# === LLM
context = "\n".join([text for text, _, _ in filtered])
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful complaints assistant. If there's not enough context, say 'I don't have enough information to answer that.'"},
        {"role": "user", "content": f"Based on the context below, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"}
    ]
)

print("\n🧠 LLM Answer:")
print(response.choices[0].message.content.strip())
