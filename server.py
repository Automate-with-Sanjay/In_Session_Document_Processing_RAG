# server.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from src.ingest import process_and_vectorize   # <-- Connects to ingestion
from src.query import answer_user_question     # <-- Connects to querying

app = FastAPI()

# Allow your UI/index.html file to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    """Endpoint for file uploads."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        process_and_vectorize(file_path, session_id)
        return {"status": "success", "message": f"{file.filename} indexed successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path) # Clean up file after embedding

@app.post("/chat")
async def chat(session_id: str = Form(...), question: str = Form(...)):
    """Endpoint for asking questions."""
    try:
        answer = answer_user_question(question, session_id)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}