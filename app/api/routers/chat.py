import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from pydantic import BaseModel

from app.api.routers.stream_response import VercelStreamResponse
from app.config import config
from app.utils.index import create_meeting_index, get_global_index, get_patient_index

chat_docs = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    patient_name: str
    prompt: str


class GlobalQuestionRequest(BaseModel):
    prompt: str


class MeetingQuestionRequest(BaseModel):
    patient_name: str
    meeting_name: str
    prompt: str


def get_query_engine(index):
    retriever = index.as_retriever(similarity_top_k=10)
    return RetrieverQueryEngine.from_args(
        retriever,
        text_qa_template=PromptTemplate(config.SYSTEM_PROMPT),
        streaming=True,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )


async def stream_response(request: Request, response):
    return VercelStreamResponse(
        request=request,
        response=response,
    )


@chat_docs.post("/ask_patient", tags=["Chat with Patient Data"])
async def chat_with_patient(request: Request, question: QuestionRequest):
    try:
        index, _ = get_patient_index(question.patient_name)
        query_engine = get_query_engine(index)
        response = await asyncio.to_thread(query_engine.query, question.prompt)
        return await stream_response(request, response)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail=str(e)
        ) from e  # Explicitly re-raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}"
        ) from e  # Explicitly re-raise


@chat_docs.post("/ask_global", tags=["Chat with All Patient Data"])
async def chat_with_all_patient_data(request: Request, question: GlobalQuestionRequest):
    logger.info(
        "Received global question: %s", question.prompt
    )  # Use lazy % formatting
    try:
        index = await asyncio.to_thread(get_global_index)
        logger.info("Global index retrieved successfully")
        query_engine = get_query_engine(index)
        logger.info("Query engine created")
        response = await asyncio.to_thread(query_engine.query, question.prompt)
        logger.info("Query executed successfully")
        return await stream_response(request, response)
    except Exception as e:
        logger.error(
            "Error in chat_with_all_patient_data: %s", str(e)
        )  # Use lazy % formatting
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}"
        ) from e  # Explicitly re-raise


@chat_docs.post("/ask_meeting", tags=["Chat with Meeting Data"])
async def chat_with_meeting(request: Request, question: MeetingQuestionRequest):
    try:
        index = create_meeting_index(question.patient_name, question.meeting_name)
        query_engine = get_query_engine(index)
        response = await asyncio.to_thread(query_engine.query, question.prompt)
        return await stream_response(request, response)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail=str(e)
        ) from e  # Explicitly re-raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}"
        ) from e  # Explicitly re-raise
