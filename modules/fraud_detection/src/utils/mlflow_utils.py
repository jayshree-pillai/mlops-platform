import mlflow

def init_mlflow():
    mlflow.set_tracking_uri("http://3.215.110.164:5000")
    print("🔗 MLflow tracking URI set to:", mlflow.get_tracking_uri())
    print("📁 MLflow experiment set to:", mlflow.get_experiment_by_name("fraud_detection_pipeline"))
