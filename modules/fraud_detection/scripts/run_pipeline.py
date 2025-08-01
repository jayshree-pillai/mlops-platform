import hydra
from omegaconf import DictConfig
from src.models.train_orchestrator import run_training
import mlflow
import mlflow.sklearn

def load_staging_model(model_name):
    model_uri = f"models:/{model_name}/staging"
    try:
        print(f"🔁 Loading model from {model_uri}")
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"⚠️  Failed to load staging model: {e}")
        return None

@hydra.main(config_path="../config", config_name="train_config.yaml")
def main(config: DictConfig):
    model = None
    if config.get("retrain_mode", False):
        model = load_staging_model(config.model_type)

    run_training(config, model)

if __name__ == "__main__":
    main()
