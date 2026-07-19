# In-Session Document Processing RAG

A FastAPI-based document ingestion and retrieval-augmented generation (RAG) app.
It lets users upload documents, vectorizes them with Google Gemini embeddings, stores them in ChromaDB,
and answers session-specific questions with Nemoguardrails safety checks.

## Key features

- Upload documents via `/upload` and build a Chroma vector store
- Query documents by `session_id` using `/chat`
- Streams ingestion logs through `/logs/stream`
- Enforces input/output safety with Nemoguardrails
- Uses Google Generative AI via `langchain-google-genai`

## Project structure

- `app.py` — FastAPI server and endpoints
- `src/ingest.py` — document ingestion, splitting, embedding, and Chroma persistence
- `src/query.py` — session-aware retrieval and Gemini-based answer generation
- `src/query_guardrails.py` — safety checks before and after answer generation
- `src/loader.py` — document loader abstraction for PDF, Word, text, and Excel files
- `guardrails/query_policy.yml` — Nemoguardrails safety policy
- `UI/index.html` — front-end interface for the chat UI

## Environment

Create a `.env` file in the repository root with:

```env
GOOGLE_API_KEY=your-google-api-key
DATA_DIR=./data
```

> `.env` is ignored by `.gitignore` and should not be committed to git.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run locally

```bash
uvicorn app:app --reload
```

Then open `UI/index.html` in a browser or connect a client to the API.

## API endpoints

- `POST /upload` — upload a file and index it for a session
- `POST /chat` — ask a question for a session
- `GET /logs/stream` — stream log events for a session

## Notes

- `.env` is not committed and should remain private. Replace `.env.example` to `.env` and update with your credentials.
- The application requires a valid `GOOGLE_API_KEY`.

Thanks !
@author - `Sanjaykumar Venkatesan`
