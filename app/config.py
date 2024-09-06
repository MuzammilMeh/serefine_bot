import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev')
    STORAGE_DIR = "./chroma_db"
    PATIENT_DATA_DIR = "./patient_data"
    BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
    EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'

config = Config()
