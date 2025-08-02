FROM python:3.11-slim

RUN pip install --no-cache-dir scikit-learn prometheus_client
#run this docker image from modules/fraud_detection/
# Copy full repo context into the container
COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY src/sagemaker_container/inference.py ./inference.py

ENV MODEL_VERSION=v1
ENV PUSHGATEWAY_URL=http://52.22.35.87:9091

ENTRYPOINT ["python", "inference.py"]
