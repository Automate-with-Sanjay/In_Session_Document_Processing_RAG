# src/ingest.py
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Check your .env file.")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.loader import load_document

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()

DB_DIR = str(DATA_DIR / "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

def process_and_vectorize(file_path: str, session_id: str, logger):
    logger.info("Loading document")
    
    documents = load_document(file_path, logger)
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    logger.info(f"Created {len(docs)} chunks")

    for doc in docs:
        doc.metadata["session_id"] = session_id
        doc.metadata["source_file"] = os.path.basename(file_path)

    logger.info("Initializing embedding model")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=API_KEY
    )

    logger.info("Embedding model initialized")

    test = embeddings.embed_query("Hello")
    logger.info(f"Embedding dimension: {len(test)}")

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    logger.success("Documents stored in ChromaDB")
    logger.success("Pipeline completed")
    logger.info("User can ask related questions now!")


    return logger.logs