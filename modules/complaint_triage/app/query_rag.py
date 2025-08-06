import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sklearn.preprocessing import normalize
import boto3
import numpy as np

BUCKET = "complaint-classifier-jp2025"
PREFIX = "rag/semantic_index/"
OUT_DIR = "semantic_index"

def download_faiss_from_s3():
    s3 = boto3.client("s3")
    os.makedirs(OUT_DIR, exist_ok=True)
    for file in ["index.faiss", "index.pkl"]:
        path = os.path.join(OUT_DIR, file)
        s3.download_file(BUCKET, PREFIX + file, path)
# === Load FAISS index
index_dir = "semantic_index"
embedding = OpenAIEmbeddings()
download_faiss_from_s3()
db = FAISS.load_local(index_dir, embedding, allow_dangerous_deserialization=True)

print("\n🧾 Sanity check — top 2 documents in FAISS:")
doc_ids = list(db.docstore._dict.keys())[:2]
for i, doc_id in enumerate(doc_ids):
    doc = db.docstore._dict[doc_id]
    print(f"\n[{i+1}] complaint_id: {doc.metadata.get('complaint_id')}")
    print(doc.page_content[:500])

# === Accept user query
query = input("❓ Ask your question: ").strip()

query_vector = embedding.embed_query(query)
# === Search top-k from normalized FAISS index
k = 5
scores, indices = db.index.search(np.array([query_vector]), k)
print(f"\n🔥 Max similarity score: {max(scores[0]):.4f}")

# === Apply similarity threshold to filter results
THRESHOLD = 0.5
filtered = [
    (db.docstore._dict[db.index_to_docstore_id[i]], scores[0][idx])
    for idx, i in enumerate(indices[0])
    if scores[0][idx] >= THRESHOLD
]

# === Guardrail: No high-similarity matches
if not filtered:
    print("\n⚠️ No relevant context found (score below threshold). LLM will not be called.")
    print("🧠 LLM Response: I'm sorry, I couldn't find enough relevant information to answer that.")
    exit()

# === Log retrieved chunks with similarity scores
print("\n🔍 Top Matches (cosine score ≥ 0.75):")
for i, (doc, score) in enumerate(filtered):
    print(f"\n[{i+1}] Score: {score:.4f}")
    print(doc.page_content)

# === Inject top-k chunks into LLM prompt
context = "\n".join([doc.page_content for doc, _ in filtered])

from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful complaints assistant. If there's not enough context, say 'I don't have enough information to answer that.'"},
        {"role": "user", "content": f"Based on the context below, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"}
    ]
)

print("\n🧠 LLM Answer:")
print(response.choices[0].message.content.strip())
