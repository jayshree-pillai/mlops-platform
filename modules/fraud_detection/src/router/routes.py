from fastapi import APIRouter
from src.utils.model_loader import get_model_and_processor
from src.features.feature_loader import load_row_by_id, transform_row
import numpy as np

router = APIRouter()
model, processor = get_model_and_processor("v1")  # hardcoded or use env/config

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/schema")
def schema():
    return {"features": processor.feature_columns}

@router.get("/predict/{tx_id}")
def predict(tx_id: int):
    row = load_row_by_id(tx_id)
    X = transform_row(row, processor)
    pred = model.predict(X)[0]
    return {"tx_id": tx_id, "prediction": int(pred)}

@router.get("/explain/{tx_id}")
def explain(tx_id: int):
    row = load_row_by_id(tx_id)
    X = transform_row(row, processor)
    shap_values = explainer(X)
    top_features = shap_values.values[0].argsort()[-3:][::-1]
    return {"top_features": top_features.tolist()}