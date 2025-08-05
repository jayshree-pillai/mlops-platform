from fastapi import FastAPI
from modules.complaint_triage.app.routes import router

app = FastAPI()
app.include_router(router)
