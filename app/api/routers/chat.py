from fastapi.responses import StreamingResponse
from app.utils.index import get_index, get_global_index
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
# Add this import if the function exists in another module
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.index import create_meeting_index
from fastapi.responses import StreamingResponse


chat_docs = APIRouter()


class QuestionRequest(BaseModel):
    patient_name: str
    prompt: str

@chat_docs.post("/ask_patient", tags=["Chat with Patient Data"])
async def chat_with_patient(request: QuestionRequest):
    try:
        index, _ = get_index(request.patient_name)

        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=3,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        
        response = query_engine.query(request.prompt)

        async def event_generator():
            try:
                for token in response.response_gen:
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



class GlobalQuestionRequest(BaseModel):
    prompt: str

@chat_docs.post("/ask_global", tags=["Chat with All Patient Data"])
async def chat_with_all_patient_data(request: GlobalQuestionRequest):
    try:
        index = get_global_index()

        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=4,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        
        response = query_engine.query(request.prompt)

        async def event_generator():
            try:
                for token in response.response_gen:
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



class MeetingQuestionRequest(BaseModel):
    patient_name: str
    meeting_name: str
    prompt: str

@chat_docs.post("/ask_meeting", tags=["Chat with Meeting Data"])
async def chat_with_meeting(request: MeetingQuestionRequest):
    try:
        index = create_meeting_index(request.patient_name, request.meeting_name)
        
        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=3,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        
        response = query_engine.query(request.prompt)

        async def event_generator():
            try:
                for token in response.response_gen:
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Meeting data not found for patient {request.patient_name} and meeting {request.meeting_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



