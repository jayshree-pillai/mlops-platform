from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, validator
from sklearn.preprocessing import normalize
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from openai import OpenAI, RateLimitError, APIError
import boto3, os, time, hashlib, redis, json
import pickle
import numpy as np
import faiss
from pathlib import Path
import yaml

router = APIRouter()
client = OpenAI()
rds = redis.Redis(host="localhost", port=6379, db=0)

# === Constants ===
BUCKET = "complaint-classifier-jp2025"


# Read artifacts from CI-deployed dir (symlink -> versions/<VERSION>)
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/srv/rag-api/artifacts/faiss/current")
PREFIX = "rag/semantic_index/current/"    # S3 bootstrap alias

OUT_DIR = ARTIFACT_DIR                    # keep your existing var usage
INDEX_PATH = os.path.join(OUT_DIR, "index.faiss")
PCA_PATH = os.path.join(OUT_DIR, "pca.transform")
PICKLE_PATH = os.path.join(OUT_DIR, "index.pkl")

THRESHOLD = 0.35
TOP_K = 5

# === Version Tags (read from version.txt if present) ===
try:
    FAISS_VERSION = Path(os.path.join(OUT_DIR, "version.txt")).read_text().strip()
except Exception:
    FAISS_VERSION = "unknown"
CHUNK_VERSION = "token-chunk-v1.0"                # same 200/100 token chunking
PROMPT_VERSION = "prompt-v1.2"                    # same LLM prompt structure
MODEL_VERSION = "text-embedding-3-large + gpt-3.5-turbo"
PROMPT_BASE = (Path(__file__).resolve().parent / "prompt" / "versions").resolve()

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
    for file in ["index.faiss", "pca.transform", "index.pkl", "manifest.json", "version.txt"]:
        path = os.path.join(OUT_DIR, file)
        try:
            s3.download_file(BUCKET, PREFIX + file, path)
        except Exception as e:
            # bootstrap is best-effort; local deploy may already have files
            print(f"bootstrap skip {file}: {e}")

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
    prompt: str | None = None  # 'v1' | 'v2' (defaults to v1)
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

def load_prompt_by_name(name: str) -> dict:
    """
    name: 'v1' | 'v2' -> loads triage_v{name}.yml and returns dict
    keys expected: system, instruction, template, params (optional)
    """
    fname = f"triage_{name}.yml"
    path = PROMPT_BASE / fname
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r") as f:
        spec = yaml.safe_load(f)
    spec["params"] = spec.get("params", {}) or {}
    return spec

def compute_prompt_version(spec: dict) -> str:
    """Deterministic 12-char version from the prompt spec."""
    payload = json.dumps(
        {
            "system": spec.get("system", ""),
            "instruction": spec.get("instruction", ""),
            "template": spec.get("template", ""),
            "params": spec.get("params", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


### ---------- Health ---------- ###
@router.get("/health")
def health():
    have = {
        "index.faiss": os.path.exists(INDEX_PATH),
        "pca.transform": os.path.exists(PCA_PATH),
        "index.pkl": os.path.exists(PICKLE_PATH),
    }
    return {
        "status": "ok" if all(have.values()) else "degraded",
        "artifacts_present": have,
        "artifact_dir": OUT_DIR,
        "faiss_version": FAISS_VERSION,
    }

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
        prompt_name = (payload.prompt or "v1").lower()  # v1 by default

        # ----- load prompt + version
        pspec = load_prompt_by_name(prompt_name)
        pv = compute_prompt_version(pspec)

        # ----- embed
        res = client.embeddings.create(model="text-embedding-3-large", input=[query])
        query_vec = np.array(res.data[0].embedding, dtype=np.float32).reshape(1, -1)

        # match build pipeline: normalize -> PCA -> normalize
        query_vec = normalize(query_vec, axis=1)
        query_pca = pca.apply_py(query_vec)
        query_pca = normalize(query_pca, axis=1)

        # ----- retrieve
        scores, indices = index.search(query_pca, TOP_K)
        filtered = [
            (texts[i], metadatas[i], scores[0][rank])
            for rank, i in enumerate(indices[0])
            if scores[0][rank] >= THRESHOLD
        ]

        if not filtered:
            ASK_FAIL.inc()
            return {
                "llm_answer": "I don't have enough information to answer that.",
                "prompt": prompt_name,
                "prompt_version": pv,
            }

        context = "\n".join([text for text, _, _ in filtered])
        context_ids = "|".join([meta["complaint_id"] for _, meta, _ in filtered])
        RETRIEVED_CONTEXT_TOKENS.observe(len(context.split()))

        # ----- cache (include prompt_version to avoid cross-version mixing)
        cache_key = hashlib.sha256(f"{pv}::{query}::{context_ids}".encode()).hexdigest()
        cached = rds.get(cache_key)
        if cached:
            CACHE_HIT.inc()
            return {
                "llm_answer": cached.decode(),
                "prompt": prompt_name,
                "prompt_version": pv,
                "cached": True,
            }

        CACHE_MISS.inc()
        start = time.time()

        system_meta = f"[meta] retrieval={FAISS_VERSION}, chunking={CHUNK_VERSION}, prompt={pv}, model={MODEL_VERSION}"
        system_prompt = f"{pspec.get('system', '').strip()} {system_meta}".strip()

        instruction = pspec.get("instruction", "").strip()
        template = pspec.get("template", "{query}")
        user_body = template.format(context=context, query=query).strip()
        if instruction:
            user_body = instruction + "\n\n" + user_body

        response = safe_openai_call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_body}
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
            "prompt_version": pv,
            "prompt_name": prompt_name,
            "model": MODEL_VERSION,
            "token_count_est": len(context.split()) + len(query.split()),
            "latency_ms": round(elapsed * 1000, 2),
            "llm_answer": answer
        }
        print("[PROMPT_LOG]", json.dumps(version_log))

        return {
            "llm_answer": answer,
            "prompt": prompt_name,
            "prompt_version": pv,
            "cached": False
        }

    except Exception as e:
        ASK_FAIL.inc()
        raise HTTPException(status_code=500, detail=str(e))