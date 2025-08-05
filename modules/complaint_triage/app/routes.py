from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, validator
from sklearn.preprocessing import normalize
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI, RateLimitError, APIError
import boto3, os, time, hashlib, redis, json

router = APIRouter()
client = OpenAI()
rds = redis.Redis(host="localhost", port=6379, db=0)

# === Constants ===
BUCKET = "complaint-classifier-jp2025"
PREFIX = "rag/semantic_index/"
OUT_DIR = "semantic_index"
THRESHOLD = 0.75
TOP_K = 5

# === Version Tags ===
FAISS_VERSION = "faiss-v1.3-2025-08-05"
CHUNK_VERSION = "token-chunk-v1.0"
PROMPT_VERSION = "prompt-v1.2"
MODEL_VERSION = "gpt-3.5-turbo"

# === Prometheus Metrics ===
ASK_LATENCY = Histogram("rag_ask_latency_seconds", "Latency of /ask request")
ASK_COUNT = Counter("rag_ask_total", "Total /ask requests served")
ASK_FAIL = Counter("rag_ask_failed", "Failed /ask requests")
RETRIEVED_CONTEXT_TOKENS = Histogram("rag_retrieved_tokens", "Token count of retrieved context")
LLM_LATENCY = Histogram("rag_llm_latency_seconds", "Latency of OpenAI completion")
ANSWER_LENGTH = Histogram("rag_llm_answer_tokens", "Token count of LLM answers")
CACHE_HIT = Counter("rag_cache_hits", "Redis cache hits")
CACHE_MISS = Counter("rag_cache_misses", "Redis cache misses")

# === S3 Loader ===
def download_faiss_from_s3():
    s3 = boto3.client("s3")
    os.makedirs(OUT_DIR, exist_ok=True)
    for file in ["index.faiss", "index.pkl"]:
        path = os.path.join(OUT_DIR, file)
        s3.download_file(BUCKET, PREFIX + file, path)

# === Load FAISS ===
download_faiss_from_s3()
embedding = OpenAIEmbeddings()
db = FAISS.load_local(OUT_DIR, embedding, allow_dangerous_deserialization=True)

# === Request Schema ===
class AskRequest(BaseModel):
    query: str
    @validator("query")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

# === OpenAI Retry Logic ===
def safe_openai_call(messages, retries=3):
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model=MODEL_VERSION,
                messages=messages
            )
        except (RateLimitError, APIError):
            time.sleep(2 ** i)
    raise RuntimeError("OpenAI call failed after retries")

### ---------- Health ---------- ###
@router.get("/health")
def health():
    return {"status": "ok"}

### ---------- Metrics ---------- ###
@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

### ---------- RAG /ask ---------- ###
@router.post("/ask")
@ASK_LATENCY.time()
def ask(payload: AskRequest):
    ASK_COUNT.inc()
    try:
        query = payload.query
        query_vector = embedding.embed_query(query)
        norm_query = normalize([query_vector])[0]

        scores, indices = db.index.search([norm_query], k=TOP_K)
        filtered = [
            (db.docstore._dict[db.index_to_docstore_id[i]], scores[0][idx])
            for idx, i in enumerate(indices[0])
            if scores[0][idx] >= THRESHOLD
        ]

        if not filtered:
            ASK_FAIL.inc()
            return {"llm_answer": "I don't have enough information to answer that."}

        for i, (doc, score) in enumerate(filtered):
            print(f"[{i+1}] score={score:.4f} :: {doc.metadata.get('complaint_id', 'n/a')}")
            print(doc.page_content[:300])

        context = "\n".join([doc.page_content for doc, _ in filtered])
        context_ids = "|".join([doc.metadata.get("complaint_id", f"doc{i}") for i, (doc, _) in enumerate(filtered)])
        RETRIEVED_CONTEXT_TOKENS.observe(len(context.split()))

        cache_key = hashlib.sha256(f"{query}::{context_ids}".encode()).hexdigest()
        cached = rds.get(cache_key)
        if cached:
            CACHE_HIT.inc()
            return {"llm_answer": cached.decode()}

        CACHE_MISS.inc()

        start = time.time()
        system_prompt = f"You are a helpful complaints assistant. [meta] retrieval={FAISS_VERSION}, chunking={CHUNK_VERSION}, prompt={PROMPT_VERSION}, model={MODEL_VERSION}"
        response = safe_openai_call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ])
        elapsed = time.time() - start
        LLM_LATENCY.observe(elapsed)

        answer = response.choices[0].message.content.strip()
        ANSWER_LENGTH.observe(len(answer.split()))
        rds.setex(cache_key, 3600, answer)

        version_log = {
            "query": query,
            "context_ids": context_ids.split("|"),
            "faiss_version": FAISS_VERSION,
            "chunk_version": CHUNK_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": MODEL_VERSION,
            "token_count_est": len(context.split()) + len(query.split()),
            "latency_ms": round(elapsed * 1000, 2),
            "llm_answer": answer
        }
        print("[PROMPT_LOG]", json.dumps(version_log))

        return {"llm_answer": answer}

    except Exception as e:
        ASK_FAIL.inc()
        raise HTTPException(status_code=500, detail=str(e))
