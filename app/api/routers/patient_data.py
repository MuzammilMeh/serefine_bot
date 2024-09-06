from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import json
from fastapi.responses import JSONResponse
from app.utils.index import summarize_patient_data
from pydantic import BaseModel
from llama_index.core import Document
from app.config import config

patient_data_router = APIRouter()

PATIENT_DATA_DIR = "./patient_data"  # Adjust this path as needed

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
                files = [f for f in os.listdir(patient_dir) if os.path.isfile(os.path.join(patient_dir, f))]
                patient_data.append({"patient": patient, "files": files})
        return {"patient_data": patient_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving patient data: {str(e)}")

@patient_data_router.get("/patient-meeting-data/{patient_name}/{meeting_name}", tags=["Patient Data"])
async def get_patient_meeting_data(patient_name: str, meeting_name: str):
    try:
        file_path = f"{PATIENT_DATA_DIR}/{patient_name}/{meeting_name}.json"
        with open(file_path, 'r') as file:
            meeting_data = json.load(file)
        return {"patient_name": patient_name, "meeting_name": meeting_name, "data": meeting_data}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Meeting data not found for patient {patient_name} and meeting {meeting_name}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Error decoding JSON data for patient {patient_name} and meeting {meeting_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving meeting data: {str(e)}")

@patient_data_router.post("/upload-patient-file", tags=["Patient Data"])
async def upload_patient_file(patient_name: str, file: UploadFile = File(...)):
    try:
        patient_dir = os.path.join(config.PATIENT_DATA_DIR, patient_name)
        os.makedirs(patient_dir, exist_ok=True)

        file_path = os.path.join(patient_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        return JSONResponse(content={"message": f"File uploaded successfully for patient {patient_name}"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@patient_data_router.post("/summarize-patient-file", tags=["Patient Data"])
async def summarize_patient_file(request: SummarizationRequest):
    try:
        file_path = os.path.join(config.PATIENT_DATA_DIR, request.patient_name, request.file_name)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_name}")
        
        summary_dir = os.path.join("summarize_output", request.patient_name)
        os.makedirs(summary_dir, exist_ok=True)
        summary_file_path = os.path.join(summary_dir, f"{os.path.splitext(request.file_name)[0]}_summary.txt")
        
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as summary_file:
                summary_text = summary_file.read()
        else:
            with open(file_path, 'r') as file:
                content = file.read()
            
            document = Document(text=content)
            summary = summarize_patient_data([document])
            summary_text = summary.text
            
            with open(summary_file_path, 'w') as summary_file:
                summary_file.write(summary_text)
        
        return JSONResponse(content={"summary": summary_text}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing file: {str(e)}")