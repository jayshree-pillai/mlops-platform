import logging

logger = logging.getLogger("fraud-api")

def log_shap_output(user_id, prob, shap_values):
    top_features = shap_values.values[0].argsort()[-3:][::-1]
    logger.info({
        "shap_run": True,
        "user_id": user_id,
        "prob": prob,
        "top_features": top_features.tolist()
    })

    # Optional: persist to Redis/S3 here if needed
