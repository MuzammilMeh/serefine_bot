import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from llama_index.core import Document
from pydantic import BaseModel

from app.config import config
from app.utils.index import summarize_patient_data_view

patient_data_router = APIRouter()

PATIENT_DATA_DIR = "./patient_data"


class SummarizationRequest(BaseModel):
    patient_name: str
    file_name: str


@patient_data_router.get("/patient-data", tags=["Patient Data"])
async def get_patient_data():
    try:
        patient_data = []
        for patient in os.listdir(PATIENT_DATA_DIR):
            patient_dir = os.path.join(PATIENT_DATA_DIR, patient)
            if os.path.isdir(patient_dir):
                files = [
                    f
                    for f in os.listdir(patient_dir)
                    if os.path.isfile(os.path.join(patient_dir, f))
                ]
                patient_data.append({"patient": patient, "files": files})
        return {"patient_data": patient_data}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving patient data: {str(e)}"
        ) from e  # Explicitly re-raise


@patient_data_router.get(
    "/patient-meeting-data/{patient_name}/{meeting_name}", tags=["Patient Data"]
)
async def get_patient_meeting_data(patient_name: str, meeting_name: str):
    try:
        file_path = f"{PATIENT_DATA_DIR}/{patient_name}/{meeting_name}.txt"
        with open(file_path, "r", encoding="utf-8") as file:  # Specify encoding
            meeting_data = file.read()
        return {
            "patient_name": patient_name,
            "meeting_name": meeting_name,
            "data": meeting_data,
        }
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Meeting data not found for patient {patient_name} and meeting {meeting_name}",
        ) from exc  # Explicitly re-raise
    except Exception as e:
        logging.error(
            "Error reading meeting data for %s/%s: %s",
            patient_name,
            meeting_name,
            str(e),  # Use lazy % formatting
        )
        raise HTTPException(
            status_code=500, detail=f"Error retrieving meeting data: {str(e)}"
        ) from e  # Explicitly re-raise


@patient_data_router.post("/upload-patient-file", tags=["Patient Data"])
async def upload_patient_file(patient_name: str, file: UploadFile = File(...)):
    try:
        patient_dir = os.path.join(config.PATIENT_DATA_DIR, patient_name)
        os.makedirs(patient_dir, exist_ok=True)

        file_path = os.path.join(patient_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        return JSONResponse(
            content={
                "message": f"File uploaded successfully for patient {patient_name}"
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading file: {str(e)}"
        ) from e  # Explicitly re-raise


@patient_data_router.post("/summarize-patient-file", tags=["Patient Data"])
async def summarize_patient_file(request: SummarizationRequest):
    try:
        # Check for file with and without extension
        file_path = os.path.join(
            config.PATIENT_DATA_DIR, request.patient_name, request.file_name
        )
        if not os.path.exists(file_path):
            # Try adding .txt extension if file not found
            file_path_with_ext = file_path + ".txt"
            if os.path.exists(file_path_with_ext):
                file_path = file_path_with_ext
            else:
                raise HTTPException(
                    status_code=404, detail=f"File not found: {request.file_name}"
                )

        # Rest of the function remains the same
        summary_dir = os.path.join("summarize_output", request.patient_name)
        os.makedirs(summary_dir, exist_ok=True)
        summary_file_path = os.path.join(
            summary_dir,
            f"{os.path.splitext(os.path.basename(file_path))[0]}_summary.txt",
        )

        if os.path.exists(summary_file_path):
            with open(
                summary_file_path, "r", encoding="utf-8"
            ) as summary_file:  # Specify encoding
                summary_text = summary_file.read()
        else:
            with open(file_path, "r", encoding="utf-8") as file:  # Specify encoding
                content = file.read()

            document = Document(text=content)
            summary = summarize_patient_data_view([document])
            summary_text = summary.text

            with open(
                summary_file_path, "w", encoding="utf-8"
            ) as summary_file:  # Specify encoding
                summary_file.write(summary_text)

        return JSONResponse(content={"summary": summary_text}, status_code=200)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error summarizing file: {str(e)}"
        ) from e  # Explicitly re-raise
