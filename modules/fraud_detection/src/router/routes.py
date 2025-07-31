from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from starlette.responses import Response
from prometheus_client import Histogram, Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
from src.utils.model_loader import get_model_and_processor
from src.features.feature_loader import load_row_by_id, transform_row
from src.services.redis_store import get_recent_txns, store_feature_vector
from src.services.explain import run_shap
from src.utils.hashing import hash_payload, hash_schema
import os, time, json, logging

router = APIRouter()
logger = logging.getLogger("fraud-api")
logger.setLevel(logging.INFO)

# Load model + processor
model, processor = get_model_and_processor(os.getenv("MODEL_VERSION", "v1"))
stored_schema_hash = processor.schema_hash

# Prometheus metrics
REQUEST_LATENCY = Histogram("fraud_latency_seconds", "Latency of inference")
CONFIDENCE_DIST = Histogram("fraud_confidence", "Confidence scores")
INPUT_SIZE = Gauge("fraud_input_vector_size", "Length of feature vector")
REQUEST_COUNT = Counter("fraud_requests_total", "Total requests served")

### ---------- Schema & Health ---------- ###

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/schema")
def schema():
    return {"features": processor.feature_columns}


### ---------- Inference ---------- ###

class InferencePayload(BaseModel):
    user_id: str
    features: list[float]

@router.post("/predict")
@REQUEST_LATENCY.time()
async def predict(payload: InferencePayload, background_tasks: BackgroundTasks):
    start = time.time()
    user_id = payload.user_id
    input_features = payload.features
    REQUEST_COUNT.inc()
    INPUT_SIZE.set(len(input_features))
    input_hash = hash_payload(input_features)

    # Rolling history lookup
    history = get_recent_txns(user_id, window=5)
    enriched_features = transform_row(input_features, processor, history)

    # Schema validation
    if stored_schema_hash != hash_schema(enriched_features):
        logger.warning(f"[SCHEMA_MISMATCH] user={user_id} version={model.version}")
        raise HTTPException(status_code=400, detail="Schema mismatch")

    # Predict
    prob = model.predict_proba([enriched_features])[0][1]
    CONFIDENCE_DIST.observe(prob)
    latency = round((time.time() - start) * 1000, 2)

    logger.info({
        "ts": start,
        "uid": user_id,
        "model": model.version,
        "latency_ms": latency,
        "input_hash": input_hash,
        "confidence": prob
    })
    # Push metrics to Prometheus PushGateway
    registry = CollectorRegistry()
    REQUEST_LATENCY.collect()[0].add_to_registry(registry)
    CONFIDENCE_DIST.collect()[0].add_to_registry(registry)
    INPUT_SIZE.collect()[0].add_to_registry(registry)
    REQUEST_COUNT.collect()[0].add_to_registry(registry)

    push_to_gateway(
        gateway=PUSHGATEWAY_URL,
        job="fraud_predictor",
        registry=registry
    )
    # Store features for audit / explainability
    background_tasks.add_task(store_feature_vector, user_id, enriched_features, prob, model.version)
    if prob > 0.8 or prob < 0.2:
        background_tasks.add_task(run_shap, model, enriched_features, user_id, prob)

    return {"probability": prob, "model_version": model.version}

### ---------- Explainability ---------- ###

@router.get("/explain/{tx_id}")
def explain(tx_id: int):
    row = load_row_by_id(tx_id)
    features = transform_row(row, processor)
    shap_output = run_shap(model, features)
    return {"tx_id": tx_id, "top_features": shap_output}
