import mlflow
mlflow.set_tracking_uri("file:///home/ubuntu/mlops-core/mlops-platform/mlruns")
mlflow.set_experiment("fraud_detection_pipeline")

from sklearn.metrics import classification_report, roc_auc_score
import joblib
import shap
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.features.feature_processor import FeatureProcessor

def log_and_report(model, model_name, X_val, y_val, params=None,run_source="manual"):
    print("Entered log and report in evaluate.py")
    with mlflow.start_run():
        mlflow.set_tag("run_source", run_source)
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("run_mode", "lean_test")  # vs full_sweep

        y_pred = model.predict(X_val)

        try:
            y_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
            mlflow.log_metric("val_auc", auc)
        except AttributeError:
            mlflow.log_metric("val_auc", -1)
            pass  # Model doesn't support predict_proba

        mlflow.log_params(params or {})

        report_txt = classification_report(y_val, y_pred)
        report_path = f"{model_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(report_txt)

        try:
            processor = FeatureProcessor.load("feature_processor.pkl")
            X_val_df = pd.DataFrame(X_val, columns=processor.feature_columns)
        except Exception as e:
            print(f"⚠️ Processor not found or failed: {e} — using default column names.")
            X_val_df = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(X_val.shape[1])])

        # SHAP explainability
        explainer = shap.Explainer(model, X_val_df)
        shap_values = explainer(X_val_df)
        shap.summary_plot(shap_values, X_val_df, show=False)

        summary_path = f"{model_name}_shap_summary.png"
        plt.savefig(summary_path)

        mlflow.log_artifact(report_path,artifact_path =model_name)
        mlflow.log_artifact(f"{model_name}_shap_summary.png", artifact_path=model_name)
        mlflow.sklearn.log_model(model,artifact_path = model_name)
