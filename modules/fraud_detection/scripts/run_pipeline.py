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

@hydra.main(config_path="../config", config_name="train_config.yaml", version_base=None)
def main(config: DictConfig):
    if config.get("retrain_mode", False):
        model = load_staging_model(config.model_type)
        assert config.model_type in ["logreg", "cart", "rf", "xgb"], "Invalid model_type"

        if model is None:
            raise ValueError("Retrain mode is enabled, but staging model not found.")
        run_training(config, model)
    else:
        run_training(config)

if __name__ == "__main__":
    main()
