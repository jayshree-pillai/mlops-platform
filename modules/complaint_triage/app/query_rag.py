import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# === Load FAISS index from local (assumes already downloaded from S3)
index_dir = "semantic_index"

embedding = OpenAIEmbeddings()
db = FAISS.load_local(index_dir, embedding, allow_dangerous_deserialization=True)

# === Accept user query
query = input("❓ Ask your question: ").strip()

# === Search top-k
results = db.similarity_search(query, k=3)

print("\n🔍 Top Matches:")
for i, r in enumerate(results):
    print(f"\n[{i+1}] {r.page_content}")

# === Inject into LLM
from openai import OpenAI
client = OpenAI()
context = "\n".join([r.page_content for r in results])

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful complaints assistant."},
        {"role": "user", "content": f"Based on the context below, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"}
    ]
)

print("\n🧠 LLM Answer:")
print(response.choices[0].message.content.strip())
