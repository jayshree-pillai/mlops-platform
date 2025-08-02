from flask import Flask, request, jsonify
from prometheus_client import CollectorRegistry, Histogram, Gauge, Counter, push_to_gateway
from src.utils.model_loader import get_model_and_processor
from src.features.feature_loader import transform_row
from src.utils.hashing import hash_schema
import os, traceback, logging

app = Flask(__name__)

# Setup Prometheus PushGateway integration
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
REQUEST_LATENCY = Histogram("fraud_latency_seconds", "Latency of inference")
CONFIDENCE_DIST = Histogram("fraud_confidence", "Confidence scores")
INPUT_SIZE = Gauge("fraud_input_vector_size", "Length of feature vector")
REQUEST_COUNT = Counter("fraud_requests_total", "Total requests served")

# Global model, processor, and schema hash
model = None
processor = None
stored_schema_hash = None

@app.before_first_request
def load_model():
    global model, processor, stored_schema_hash
    try:
        model, processor = get_model_and_processor(os.getenv("MODEL_VERSION", "v1"))
        stored_schema_hash = processor.schema_hash
        print("✅ Model & processor loaded.")
    except Exception as e:
        logging.error("❌ Model load failed: %s", str(e))
        traceback.print_exc()

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok" if model else "model not loaded"})

@app.route("/invocations", methods=["POST"])
@REQUEST_LATENCY.time()
def invoke():
    try:
        REQUEST_COUNT.inc()
        data = request.get_json()
        features = data["features"]
        user_id = data.get("user_id", "unknown")

        INPUT_SIZE.set(len(features))
        transformed = transform_row(features, processor)

        if hash_schema(transformed) != stored_schema_hash:
            return jsonify({"error": "schema hash mismatch"}), 400

        proba = model.predict_proba([transformed])[0][1]
        CONFIDENCE_DIST.observe(proba)

        # Push custom metrics
        registry = CollectorRegistry()
        for metric in [REQUEST_LATENCY, CONFIDENCE_DIST, INPUT_SIZE, REQUEST_COUNT]:
            metric.collect()[0].add_to_registry(registry)
        push_to_gateway(PUSHGATEWAY_URL, job="fraud-byoc", registry=registry)

        return jsonify({
            "probability": proba,
            "model_version": model.version,
            "status": "success"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
