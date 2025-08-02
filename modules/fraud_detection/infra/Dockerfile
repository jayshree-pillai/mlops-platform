FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#run this docker image from modules/fraud_detection/
COPY src ./src
COPY scripts ./scripts
COPY config ./config

EXPOSE 8080

CMD ["uvicorn", "src.router.app:app", "--host", "0.0.0.0", "--port", "8080"]
