# src/loader.py
import os
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
)

def load_document(file_path: str):
    """Detects file type and extracts raw text documents."""
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.pdf':
        return PyPDFLoader(file_path).load()
    elif ext in ['.docx', '.doc']:
        return Docx2txtLoader(file_path).load()
    elif ext == '.txt':
        return TextLoader(file_path, encoding='utf-8').load()
    elif ext in ['.xlsx', '.xls']:
        return UnstructuredExcelLoader(file_path, mode="elements").load()
    else:
        raise ValueError(f"Unsupported file format: {ext}")