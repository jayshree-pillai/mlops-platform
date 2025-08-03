import mlflow

def init_mlflow(tracking_uri="http://127.0.0.1:5000", experiment_name="fraud_default"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print("🔗 Tracking URI:", mlflow.get_tracking_uri())
    print("📁 Experiment:", mlflow.get_experiment_by_name(experiment_name))
