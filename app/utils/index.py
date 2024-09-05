import logging
import os
import json
# from features.docs.dependency import get_file_path
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceWindowNodeParser, SimpleNodeParser
from llama_index.llms.openai import OpenAI

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import logging
from sqlalchemy.orm import Session
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = "sk-proj-H1GNnjxf1oq2JzHc1r6gT3BlbkFJX6yPVR204SYTFRJCkjNT"
llm = OpenAI(model="gpt-4-1106-preview", temperature=0)
#1106
STORAGE_DIR = "./chroma_db"  # directory to cache the generated index

PATIENT_DATA_DIR = "./patient_data"

embed_model=HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')

Settings.embed_model=embed_model
Settings.llm = llm

def get_patient_index(patient_name):
    patient_dir = os.path.join(PATIENT_DATA_DIR, patient_name)
    if not os.path.isdir(patient_dir):
        raise FileNotFoundError(f"Patient directory not found: {patient_name}")

    collection_name = f"{patient_name}_collection"
    db = chromadb.PersistentClient(path=STORAGE_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if len(chroma_collection.get()["documents"]) == 0:
        # Use SimpleDirectoryReader to load all files in the patient's directory and subdirectories
        reader = SimpleDirectoryReader(
            input_dir=patient_dir,
            recursive=True
        )
        documents = reader.load_data()

        node_parser = SimpleNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(documents)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

    return index, collection_name

def get_index(patient_name):
    return get_patient_index(patient_name)



def get_global_index():
    collection_name = "global_patient_data"
    db = chromadb.PersistentClient(path=STORAGE_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if len(chroma_collection.get()["documents"]) == 0:
        all_documents = []
        for patient in os.listdir(PATIENT_DATA_DIR):
            patient_dir = os.path.join(PATIENT_DATA_DIR, patient)
            if os.path.isdir(patient_dir):
                reader = SimpleDirectoryReader(input_dir=patient_dir, recursive=True)
                patient_documents = reader.load_data()
                patient_summary = summarize_patient_data(patient_documents)
                all_documents.extend(patient_documents + [patient_summary])

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=10,  # Increase this value
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = node_parser.get_nodes_from_documents(all_documents)

        index = VectorStoreIndex(
            nodes,
            storage_context=StorageContext.from_defaults(vector_store=vector_store)
        )
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

    return index


def summarize_patient_data(documents):
    
    summary_template = (
        "Summarize the key points and trends from the following patient data:\n"
        "{text}\n"
        "Summary:"
    )
    summary = llm.predict(summary_template.format(text="\n".join([doc.text for doc in documents])))
    return Document(text=summary, extra_info={"type": "patient_summary"})

def create_meeting_index(patient_name: str, meeting_name: str):
    file_path = os.path.join(PATIENT_DATA_DIR, patient_name, f"{meeting_name}.json")
    collection_name = f"{patient_name}_{meeting_name}"

    db = chromadb.PersistentClient(path=STORAGE_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if len(chroma_collection.get()["documents"]) == 0:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert the JSON data to a string
        text = json.dumps(data)
        
        # Create a Document object
        document = Document(text=text)
        
        # Create nodes from the document
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=10,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = node_parser.get_nodes_from_documents([document])
        
        # Create an index and store it in the vector database
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    else:
        # Load the existing index from the vector store
        index = VectorStoreIndex.from_vector_store(vector_store)

    return index
