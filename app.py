# app.py
import asyncio
import json
import os
import traceback
from typing import Dict, Set
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()



from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.ingest import process_and_vectorize
from src.query_guardrails import generate_guarded_response
from src.logger import PipelineLogger
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = DATA_DIR / "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

session_log_queues: Dict[str, Set[asyncio.Queue]] = {}


def publish_log(session_id: str, level: str, message: str) -> None:
    payload = {
        "type": "log",
        "level": level,
        "message": message,
    }

    for queue in list(session_log_queues.get(session_id, set())):
        queue.put_nowait(payload)

@app.get("/logs/stream")
async def logs_stream(request: Request, session_id: str):
    queue = asyncio.Queue()
    session_log_queues.setdefault(session_id, set()).add(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(item)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            session_log_queues.get(session_id, set()).discard(queue)
            if session_id in session_log_queues and not session_log_queues[session_id]:
                session_log_queues.pop(session_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    def on_log(level: str, message: str):
        publish_log(session_id, level, message)

    logger = PipelineLogger(callback=on_log)


    try:
        logs = await asyncio.to_thread(process_and_vectorize, file_path, session_id, logger)
        return {
            "status": "success",
            "message": f"{file.filename} indexed successfully.",
            "logs": logs,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "logs": [str(e)],
        }
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                publish_log(session_id, "DONE", "Finished")
            except PermissionError:
                pass


@app.post("/chat")
async def chat(session_id: str = Form(...), question: str = Form(...)):
    print (f"Received question: {question} for session_id: {session_id}")
    if not question or not question.strip():
        return {"status": "error", "message": "Question cannot be empty."}

    if len(question.strip()) > 2000:
        return {"status": "error", "message": "Question is too long; please shorten it."}

    try:
        answer = await generate_guarded_response(question, session_id)
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}