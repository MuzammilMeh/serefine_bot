import logging
import os
import json
from typing import List, Tuple

from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceWindowNodeParser, SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.bedrock import Bedrock

import boto3
from botocore.config import Config
import botocore
import chromadb
import asyncio

from app.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_bedrock_client():
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=config.AWS_DEFAULT_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            config=Config(retries={'max_attempts': 10, 'mode': 'standard'})
        )
        logger.info(f"Bedrock client initialized with region: {config.AWS_DEFAULT_REGION}")
        return bedrock_runtime
    except botocore.exceptions.ClientError as e:
        logger.error(f"Error accessing Bedrock: {e}")
        handle_bedrock_error(e)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    return None

def handle_bedrock_error(e):
    if e.response['Error']['Code'] == 'AccessDeniedException':
        logger.error("You don't have access to Bedrock. Please check your AWS credentials and permissions.")
    elif e.response['Error']['Code'] == 'UnrecognizedClientException':
        logger.error("Invalid AWS credentials. Please check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
    else:
        logger.error(f"Unexpected error: {e}")

def initialize_llm(bedrock_runtime):
    try:
        llm = Bedrock(
            model=config.BEDROCK_MODEL,
            client=bedrock_runtime,
            context_size=200000,
            temperature=0,
            
        )
        logger.info(f"Successfully initialized Bedrock LLM with model: {llm.model}")
        
        return llm
    except Exception as e:
        logger.error(f"Error initializing Bedrock LLM: {str(e)}")
        if isinstance(e, botocore.exceptions.ClientError):
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS Error Code: {error_code}")
            logger.error(f"AWS Error Message: {error_message}")
    return None

def initialize_settings():
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    Settings.llm = initialize_llm(initialize_bedrock_client())

initialize_settings()

def get_patient_index(patient_name: str) -> Tuple[VectorStoreIndex, str]:
    patient_dir = os.path.join(config.PATIENT_DATA_DIR, patient_name)
    if not os.path.isdir(patient_dir):
        raise FileNotFoundError(f"Patient directory not found: {patient_name}")

    collection_name = f"{patient_name}_collection"
    db = chromadb.PersistentClient(path=config.STORAGE_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if len(chroma_collection.get()["documents"]) == 0:
        documents = SimpleDirectoryReader(input_dir=patient_dir, recursive=True).load_data()
        nodes = SimpleNodeParser.from_defaults().get_nodes_from_documents(documents)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

    return index, collection_name

def get_global_index() -> VectorStoreIndex:
    logger.info("Starting get_global_index")
    collection_name = "global_patient_data"
    db = chromadb.PersistentClient(path=config.STORAGE_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    try:
        if len(chroma_collection.get()["documents"]) == 0:
            logger.info("Creating new global index")
            all_documents = []
            for patient in os.listdir(config.PATIENT_DATA_DIR):
                patient_dir = os.path.join(config.PATIENT_DATA_DIR, patient)
                if os.path.isdir(patient_dir):
                    logger.info(f"Processing patient directory: {patient}")
                    patient_documents = SimpleDirectoryReader(input_dir=patient_dir, recursive=True).load_data()
                    logger.info(f"Loaded {len(patient_documents)} documents for patient {patient}")
                    patient_summary = summarize_patient_data_view(patient_documents)
                    logger.info(f"Created summary for patient {patient}")
                    all_documents.extend(patient_documents + [patient_summary])
            
            logger.info(f"Total documents loaded: {len(all_documents)}")
            logger.info("Creating nodes from documents")
            nodes = SentenceWindowNodeParser.from_defaults(
                window_size=10,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            ).get_nodes_from_documents(all_documents)
            logger.info(f"Created {len(nodes)} nodes")

            logger.info("Creating VectorStoreIndex")
            index = VectorStoreIndex(
                nodes,
                storage_context=StorageContext.from_defaults(vector_store=vector_store)
            )
            logger.info("New global index created successfully")
        else:
            logger.info("Loading existing global index")
            index = VectorStoreIndex.from_vector_store(vector_store)
            logger.info("Existing global index loaded successfully")

        return index
    except Exception as e:
        logger.error(f"Error in get_global_index: {str(e)}")
        raise

def create_meeting_index(patient_name: str, meeting_name: str) -> VectorStoreIndex:
    file_path = os.path.join(config.PATIENT_DATA_DIR, patient_name, f"{meeting_name}.txt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Meeting file not found: {file_path}")
    
    # Sanitize the collection name
    sanitized_meeting_name = meeting_name.replace(" ", "_").lower()
    collection_name = f"{patient_name.lower()}_{sanitized_meeting_name}"[:63]  # Limit to 63 characters
    
    db = chromadb.PersistentClient(path=config.STORAGE_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if len(chroma_collection.get()["documents"]) == 0:
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        nodes = SimpleNodeParser.from_defaults().get_nodes_from_documents(documents)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

    return index

def summarize_patient_data(documents: List[Document]) -> Document:
    summary_template = """You are an AI Psychologist with a specialty of diagnosing Autism in children.
    The following is a transcription of a conversation between our Staff and one or more parent/guardian of a child with Autism, and may even include a translator - You will need to determine the speakers.
    Your job is to review the transcription and provide an extremely detailed summary of the conversation that includes as much a detail as possible.


    {text}

    Summary:"""
    
    chunk_size = 4000  # Adjust this value based on the model's input limit
    summaries = []
    
    for i in range(0, len(documents), 3):  # Process 3 documents at a time
        chunk_docs = documents[i:i+3]
        text_content = "\n".join([doc.text for doc in chunk_docs])
        
        if len(text_content) > chunk_size:
            # Split the text content into smaller chunks
            chunks = [text_content[j:j+chunk_size] for j in range(0, len(text_content), chunk_size)]
            chunk_summaries = []
            
            for chunk in chunks:
                chunk_summary = Settings.llm.complete(summary_template.format(text=chunk))
                chunk_summaries.append(chunk_summary.text)
            
            summaries.append("\n".join(chunk_summaries))
        else:
            summary = Settings.llm.complete(summary_template.format(text=text_content))
            summaries.append(summary.text)
    
    # Combine all summaries
    final_summary = "\n\n".join(summaries)
    
    # If the final summary is still too long, summarize it again
    if len(final_summary) > chunk_size:
        final_summary = Settings.llm.complete(summary_template.format(text=final_summary)).text
    
    return Document(text=final_summary, extra_info={"type": "patient_summary"})



def summarize_patient_data_view(documents: List[Document]) -> Document:
    summary_template = """You are an AI Psychologist with a specialty of diagnosing Autism in children.
The following is a transcription of a conversation between our Staff and one or more parent/guardian of a child with Autism, and may even include a translator - You will need to determine the speakers.
Your job is to review the transcription and provide an extremely detailed summary of the conversation that includes as much a detail as possible.

{text}

Summary:"""
    
    text_content = "\n".join([doc.text for doc in documents])
    summary = Settings.llm.complete(summary_template.format(text=text_content))
    return Document(text=summary.text, extra_info={"type": "patient_summary"})