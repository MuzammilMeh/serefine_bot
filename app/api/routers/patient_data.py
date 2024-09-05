from fastapi import APIRouter, HTTPException
import os
import json

patient_data_router = APIRouter()

PATIENT_DATA_DIR = "./patient_data"  # Adjust this path as needed

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