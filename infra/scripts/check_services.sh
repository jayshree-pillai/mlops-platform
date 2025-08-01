#!/bin/bash

echo "🔍 Checking Redis..."
sudo systemctl is-active --quiet redis-server && echo "✅ Redis is running" || echo "❌ Redis NOT running"

echo "🔍 Checking MLflow..."
MLFLOW_UP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000)
[[ "$MLFLOW_UP" == "200" ]] && echo "✅ MLflow is running" || echo "❌ MLflow NOT running"

#echo "🔍 Checking Prometheus..."
#PROM_UP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/graph)
#[[ "$PROM_UP" == "200" ]] && echo "✅ Prometheus is running" || echo "❌ Prometheus NOT running"

#echo "🔍 Checking Grafana..."
#GRAFANA_UP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/login)
#[[ "$GRAFANA_UP" == "200" ]] && echo "✅ Grafana is running" || echo "❌ Grafana NOT running"

echo "🔍 Checking Docker image..."
docker images | grep fraud-env &>/dev/null && echo "✅ fraud-env Docker image exists" || echo "❌ fraud-env Docker image MISSING"