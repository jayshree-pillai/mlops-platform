import mlflow
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def log_and_report(model, model_name, X_val, y_val, params=None):
    y_pred = model.predict(X_val)

    try:
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        mlflow.log_metric("val_auc", auc)
    except AttributeError:
        pass  # Model doesn't support predict_proba

    mlflow.log_params(params or {})

    report_txt = classification_report(y_val, y_pred)
    report_path = f"{model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(report_txt)
    mlflow.log_artifact(report_path)

    mlflow.sklearn.log_model(model, "model")
