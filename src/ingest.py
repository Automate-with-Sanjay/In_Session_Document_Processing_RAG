# src/ingest.py
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.loader import load_document  # <-- Connects to loader

DB_DIR = "./chroma_db"

def process_and_vectorize(file_path: str, session_id: str):
    """Extracts, chunks, tags with session_id, and indexes the document."""
    # 1. Read file text
    documents = load_document(file_path)
    
    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # 3. Add session isolation tags
    for doc in docs:
        doc.metadata["session_id"] = session_id
        doc.metadata["source_file"] = os.path.basename(file_path)
        
    # 4. Save to Chroma DB
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=OpenAIEmbeddings(),
        persist_directory=DB_DIR
    )
    return vector_store