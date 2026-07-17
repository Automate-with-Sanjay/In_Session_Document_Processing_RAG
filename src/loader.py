# src/loader.py
import os
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
)

def load_document(file_path: str, logger):
    """Detects file type and extracts raw text documents."""
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.pdf':
        logger.info("Loading PDF document")
        return PyPDFLoader(file_path).load()
    
    elif ext in ['.docx', '.doc']:
        logger.info("Loading Word document")
        return Docx2txtLoader(file_path).load()
    elif ext == '.txt':
        logger.info("Loading text document")
        return TextLoader(file_path, encoding='utf-8').load()
    elif ext in ['.xlsx', '.xls']:
        logger.info("Loading Excel document")
        return UnstructuredExcelLoader(file_path, mode="elements").load()
    else:
        logger.error(f"Unsupported file format: {ext}")
        raise ValueError(f"Unsupported file format: {ext}")
