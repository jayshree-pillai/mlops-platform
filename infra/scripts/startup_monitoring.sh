#!/bin/bash

set -e

echo "🔧 Installing dependencies..."
sudo apt-get update
sudo apt-get install -y wget curl tar

### === PROMETHEUS === ###
echo "📦 Installing Prometheus..."
cd /opt
sudo useradd --no-create-home --shell /bin/false prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.51.2/prometheus-2.51.2.linux-amd64.tar.gz
tar -xzf prometheus-2.51.2.linux-amd64.tar.gz
sudo mv prometheus-2.51.2.linux-amd64 prometheus
rm prometheus-2.51.2.linux-amd64.tar.gz

### === PUSHGATEWAY === ###
echo "📦 Installing PushGateway..."
wget https://github.com/prometheus/pushgateway/releases/download/v1.7.0/pushgateway-1.7.0.linux-amd64.tar.gz
tar -xzf pushgateway-1.7.0.linux-amd64.tar.gz
sudo mv pushgateway-1.7.0.linux-amd64 pushgateway
rm pushgateway-1.7.0.linux-amd64.tar.gz

### === GRAFANA === ###
echo "📦 Installing Grafana..."
sudo apt-get install -y apt-transport-https software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
sudo apt-get install -y gnupg2
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install -y grafana

### === Start Services === ###
echo "🚀 Starting Prometheus, PushGateway, Grafana..."
nohup /opt/prometheus/prometheus --config.file=/opt/prometheus/prometheus.yml > /home/ubuntu/prometheus.log 2>&1 &
nohup /opt/pushgateway/pushgateway > /home/ubuntu/pushgateway.log 2>&1 &
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

echo "✅ Monitoring stack is live."
