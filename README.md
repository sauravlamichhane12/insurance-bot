# MediLink Insurance Assistant

MediLink Insurance Assistant is a FastAPI application for insurance document ingestion and document-grounded chat. Admins upload PDF, CSV, or XLSX files, and signed-in users can ask questions against the indexed content.

## Features

- Local username/password login for users and admins
- Admin dashboard for user management and document ingestion
- PDF, CSV, and native XLSX indexing
- Streaming chat responses over Server-Sent Events
- SQLite-backed session and user storage
- Pinecone + AWS Bedrock retrieval pipeline

## Stack

- FastAPI
- SQLite
- AWS Bedrock
- Pinecone
- LangChain / LangGraph
- Vanilla HTML, CSS, and JavaScript

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Required environment variables:

- `SESSION_SECRET`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `PINECONE_API_KEY`
- `PINECONE_INDEX` or `PINECONE_INDEX_NAME`

Optional bootstrap admin variables:

- `AUTH_USERNAME`
- `AUTH_PASSWORD`

## Document Flow

1. Admin uploads a PDF, CSV, or XLSX document.
2. The app validates and stores the file locally.
3. The RAG service extracts text or row-level records and indexes them in Pinecone.
4. Signed-in users ask questions through the chat UI.
5. The assistant answers from the uploaded MediLink insurance documents only.

## Notes

- XLSX ingestion uses the built-in workbook reader and preserves worksheet order and row sequence.
- The included `Health Benefits (Peso Plans).xlsx` file is supported directly by the ingestion pipeline.
