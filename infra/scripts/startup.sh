#!/bin/bash

# === Activate virtualenv ===
source /home/ubuntu/mlops-core/venv/bin/activate

# === Navigate to repo and pull latest ===
cd /home/ubuntu/mlops-core/mlops-platform
git pull origin main


# === Ensure required packages ===
pip install -r /home/ubuntu/mlops-core/mlops-platform/requirements.txt


# === Build Docker image for ML environment ===
docker build -t fraud-env -f infra/Dockerfile.env_setup .

# === Start Redis ===
sudo systemctl start redis-server

# === Ensure PostgreSQL container is running ===
docker update --restart=always mlflow-postgres
docker start mlflow-postgres 2>/dev/null || true

until pg_isready -h localhost -p 5432 -U mlflow_user; do
  echo "Waiting for Postgres..."
  sleep 2
done

# === Start MLflow Tracking Server ===
mlflow ui --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db --default-artifact-root s3://mlops-fraud-dev/models/ --host 0.0.0.0 --port 5000 &

# === Start Prometheus ===
#nohup /home/ubuntu/mlops-core/prometheus/prometheus --config.file=/home/ubuntu/mlops-core/prometheus/prometheus.yml > /home/ubuntu/prometheus.log 2>&1 &

# === Start Grafana ===
#sudo systemctl start grafana-server