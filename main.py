import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.chat import chat_docs
from app.api.routers.patient_data import patient_data_router
from app.arize_client import setup_arize_client
from app.config import config
from app.observability import init_observability
from app.utils.error_handler import http_error_handler

# init_observability()
setup_arize_client()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(HTTPException, http_error_handler)
app.include_router(chat_docs)
app.include_router(patient_data_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        # workers=4,
    )
