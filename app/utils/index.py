import logging
import os
from typing import List, Tuple

import boto3
import botocore
import chromadb
from botocore.config import Config
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.bedrock import Bedrock
from llama_index.vector_stores.chroma import ChromaVectorStore

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
            config=Config(retries={"max_attempts": 10, "mode": "standard"}),
        )
        logger.info(
            "Bedrock client initialized with region: %s",
            config.AWS_DEFAULT_REGION,  # Use lazy % formatting
        )
        return bedrock_runtime
    except botocore.exceptions.ClientError as e:
        logger.error("Error accessing Bedrock: %s", e)  # Use lazy % formatting
        handle_bedrock_error(e)
    except (
        boto3.exceptions.Boto3Error,
        botocore.exceptions.BotoCoreError,
    ) as e:  # Catch more specific exceptions
        logger.error("Unexpected error: %s", e)  # Use lazy % formatting
    return None


def handle_bedrock_error(e):
    if e.response["Error"]["Code"] == "AccessDeniedException":
        logger.error(
            "You don't have access to Bedrock. Please check your AWS credentials and permissions."
        )
    elif e.response["Error"]["Code"] == "UnrecognizedClientException":
        logger.error(
            "Invalid AWS credentials. Please check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )
    else:
        logger.error("Unexpected error: %s", e)  # Use lazy % formatting


def initialize_llm(bedrock_runtime):
    try:
        llm = Bedrock(
            model=config.BEDROCK_MODEL,
            client=bedrock_runtime,
            context_size=200000,
            temperature=0,
        )
        logger.info(
            "Successfully initialized Bedrock LLM with model: %s", llm.model
        )  # Use lazy % formatting

        return llm
    except (
        botocore.exceptions.ClientError,
        boto3.exceptions.Boto3Error,
    ) as e:  # Catch more specific exceptions
        logger.error(
            "Error initializing Bedrock LLM: %s", str(e)
        )  # Use lazy % formatting
        if isinstance(e, botocore.exceptions.ClientError):
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error("AWS Error Code: %s", error_code)  # Use lazy % formatting
            logger.error(
                "AWS Error Message: %s", error_message
            )  # Use lazy % formatting
    return None


def initialize_settings():
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    Settings.llm = initialize_llm(initialize_bedrock_client())

    # os.environ["OPENAI_API_KEY"] = "sk-proj-H1GNnjxf1oq2JzHc1r6gT3BlbkFJX6yPVR204SYTFRJCkjNT"
    # llm = OpenAI(model="gpt-4-1106-preview", temperature=0)
    # Settings.llm = llm


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
        documents = SimpleDirectoryReader(
            input_dir=patient_dir, recursive=True
        ).load_data()
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
                    logger.info(
                        "Processing patient directory: %s", patient
                    )  # Use lazy % formatting
                    patient_documents = SimpleDirectoryReader(
                        input_dir=patient_dir, recursive=True
                    ).load_data()
                    logger.info(
                        "Loaded %d documents for patient %s",
                        len(patient_documents),
                        patient,  # Use lazy % formatting
                    )
                    patient_summary = summarize_patient_data_view(patient_documents)
                    logger.info(
                        "Created summary for patient %s", patient
                    )  # Use lazy % formatting
                    all_documents.extend(patient_documents + [patient_summary])

            logger.info(
                "Total documents loaded: %d", len(all_documents)
            )  # Use lazy % formatting
            logger.info("Creating nodes from documents")
            nodes = SimpleNodeParser.from_defaults().get_nodes_from_documents(
                all_documents
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
            logger.info("New global index created successfully")
        else:
            logger.info("Loading existing global index")
            index = VectorStoreIndex.from_vector_store(vector_store)
            logger.info("Existing global index loaded successfully")

        return index
    except Exception as e:
        logger.error("Error in get_global_index: %s", str(e))  # Use lazy % formatting
        raise


def create_meeting_index(patient_name: str, meeting_name: str) -> VectorStoreIndex:
    file_path = os.path.join(
        config.PATIENT_DATA_DIR, patient_name, f"{meeting_name}.txt"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Meeting file not found: {file_path}")

    # Sanitize the collection name
    sanitized_meeting_name = meeting_name.replace(" ", "_").lower()
    collection_name = f"{patient_name.lower()}_{sanitized_meeting_name}"[
        :63
    ]  # Limit to 63 characters

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
        chunk_docs = documents[i : i + 3]
        text_content = "\n".join([doc.text for doc in chunk_docs])

        if len(text_content) > chunk_size:
            # Split the text content into smaller chunks
            chunks = [
                text_content[j : j + chunk_size]
                for j in range(0, len(text_content), chunk_size)
            ]
            chunk_summaries = []

            for chunk in chunks:
                chunk_summary = Settings.llm.complete(
                    summary_template.format(text=chunk)
                )
                chunk_summaries.append(chunk_summary.text)

            summaries.append("\n".join(chunk_summaries))
        else:
            summary = Settings.llm.complete(summary_template.format(text=text_content))
            summaries.append(summary.text)

    # Combine all summaries
    final_summary = "\n\n".join(summaries)

    # If the final summary is still too long, summarize it again
    if len(final_summary) > chunk_size:
        final_summary = Settings.llm.complete(
            summary_template.format(text=final_summary)
        ).text

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
