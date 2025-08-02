FROM python:3.9

RUN pip install --no-cache-dir scikit-learn prometheus_client

COPY inference.py ./
COPY model_loader.py ./

ENV MODEL_VERSION=v1
ENV PUSHGATEWAY_URL=http://52.22.35.87:9091

ENTRYPOINT ["python", "inference.py"]
