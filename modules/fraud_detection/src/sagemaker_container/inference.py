import json, time, os
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from model_loader import load_model  # your existing logic

model, processor = load_model(os.getenv("MODEL_VERSION", "v1"))
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL")

def handler(data):
    start = time.time()
    features = data["features"]

    enriched = processor.transform(features)
    prob = model.predict_proba([enriched])[0][1]

    # Push metrics
    registry = CollectorRegistry()
    Gauge("byoc_latency_ms", "Latency", registry=registry).set((time.time() - start) * 1000)
    Gauge("byoc_confidence", "Score", registry=registry).set(prob)
    push_to_gateway(PUSHGATEWAY_URL, job="sagemaker_byoc", registry=registry)

    return {"probability": prob}

def predict_fn(input_data, context):
    return handler(json.loads(input_data))

def input_fn(input_data, content_type):
    return input_data.decode("utf-8")

def output_fn(prediction, accept):
    return json.dumps(prediction), accept
