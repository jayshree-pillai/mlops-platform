from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, validator
from sklearn.preprocessing import normalize
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from openai import OpenAI, RateLimitError, APIError
import boto3, os, time, hashlib, redis, json
import pickle
import numpy as np
import faiss

router = APIRouter()
client = OpenAI()
rds = redis.Redis(host="localhost", port=6379, db=0)

# === Constants ===
BUCKET = "complaint-classifier-jp2025"
PREFIX = "rag/semantic_index/"
OUT_DIR = "semantic_index"
INDEX_PATH = f"{OUT_DIR}/index.faiss"
PCA_PATH = f"{OUT_DIR}/pca.transform"
PICKLE_PATH = f"{OUT_DIR}/index.pkl"
THRESHOLD = 0.35
TOP_K = 5

# === Version Tags ===
FAISS_VERSION = "faiss-pca-v2.0-2025-08-07"       # uses PCA + cosine norm
CHUNK_VERSION = "token-chunk-v1.0"                # same 200/100 token chunking
PROMPT_VERSION = "prompt-v1.2"                    # same LLM prompt structure
MODEL_VERSION = "text-embedding-3-large + gpt-3.5-turbo"


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

# === Load FAISS + PCA + Metadata ===
download_faiss_from_s3()
index = faiss.read_index(INDEX_PATH)
pca = faiss.read_VectorTransform(PCA_PATH)
with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)
texts = data["texts"]
metadatas = data["metadata"]

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
                model="gpt-3.5-turbo",
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
        res = client.embeddings.create(model="text-embedding-3-large", input=[query])
        query_vec = np.array(res.data[0].embedding, dtype=np.float32).reshape(1, -1)
        query_pca = pca.apply_py(query_vec)
        query_pca = normalize(query_pca, axis=1)

        scores, indices = index.search(query_pca, TOP_K)
        filtered = [
            (texts[i], metadatas[i], scores[0][rank])
            for rank, i in enumerate(indices[0])
            if scores[0][rank] >= THRESHOLD
        ]

        if not filtered:
            ASK_FAIL.inc()
            return {"llm_answer": "I don't have enough information to answer that."}

        context = "\n".join([text for text, _, _ in filtered])
        context_ids = "|".join([meta["complaint_id"] for _, meta, _ in filtered])
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