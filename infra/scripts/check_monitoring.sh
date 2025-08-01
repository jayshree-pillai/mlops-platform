#!/bin/bash

echo "🔍 Checking Prometheus..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/graph | grep 200 &>/dev/null && echo "✅ Prometheus is running" || echo "❌ Prometheus NOT running"

echo "🔍 Checking PushGateway..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:9091 | grep 200 &>/dev/null && echo "✅ PushGateway is running" || echo "❌ PushGateway NOT running"

echo "🔍 Checking Grafana..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/login | grep 200 &>/dev/null && echo "✅ Grafana is running" || echo "❌ Grafana NOT running"
