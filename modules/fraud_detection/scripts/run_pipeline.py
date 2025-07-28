import hydra
from omegaconf import DictConfig
from src.models.train_orchestrator import run_training

@hydra.main(config_path="../config", config_name="train_config.yaml")
def main(config: DictConfig):
    run_training(config)

if __name__ == "__main__":
    main()
