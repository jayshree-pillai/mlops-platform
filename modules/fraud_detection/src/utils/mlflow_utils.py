import mlflow

def init_mlflow():
    mlflow.set_tracking_uri("file:///home/ubuntu/mlops-core/mlops-platform/mlruns")
    mlflow.set_experiment("fraud_detection_pipeline")
    print("🔗 MLflow tracking URI set to:", mlflow.get_tracking_uri())
    print("📁 MLflow experiment set to:", mlflow.get_experiment_by_name("fraud_detection_pipeline"))
