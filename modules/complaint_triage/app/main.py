from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os

app = FastAPI()
embedding = OpenAIEmbeddings()
db = FAISS.load_local("semantic_index", embedding, allow_dangerous_deserialization=True)
client = OpenAI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QueryRequest):
    query = request.question.strip()
    results = db.similarity_search(query, k=3)
    context = "\n".join([r.page_content for r in results])

    llm_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful complaints assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return {
        "answer": llm_response.choices[0].message.content.strip(),
        "matches": [r.page_content for r in results]
    }
### test it wtih
### uvicorn modules.complaint_triage.app.main:app --reload
### curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Why was I charged twice?"}'
