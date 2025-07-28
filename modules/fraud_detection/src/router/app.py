from fastapi import FastAPI
from src.router.routes import router

app = FastAPI(title="Fraud Detection API", version="1.0")
app.include_router(router)