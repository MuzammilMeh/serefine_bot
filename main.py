import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers.chat import chat_docs
from app.api.routers.patient_data import patient_data_router
from app.observability import init_observability
from app.config import config
from app.utils.error_handler import http_error_handler
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

init_observability()

app = FastAPI()

if config.ENVIRONMENT == "dev":
    logger.warning("Running in development mode - allowing CORS for all origins")
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
    uvicorn.run(app="main:app", host="0.0.0.0", reload=True)
