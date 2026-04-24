# 🛡️ MediLink Insurance Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)

**MediLink Insurance Assistant** is a professional, enterprise-ready AI assistant built to handle insurance document queries. It leverages **FastAPI**, **AWS Bedrock**, and **Pinecone** to provide grounded, secure, and real-time answers based on private insurance policies.

---

## 🌟 Key Features

### 👤 User Experience
- **Grounded Q&A:** Get answers directly from uploaded insurance PDFs and CSVs.
- **Direct Factual Responses:** Strictly factual answers without conversational preambles or filler.
- **Real-time Streaming:** Watch answers generate instantly via Server-Sent Events (SSE).
- **Persistent Memory:** Maintains context using `thread_id` via SQLite checkpointer (LangGraph) for natural follow-up questions.
- **Secure Login:** Local username/password authentication for both users and administrators.

### 🛠️ Intelligent Document Parsing
- **Triple-Tier Hybrid PDF Parsing:**
  - **Structural (Markdownification):** Converts PDF layout to Markdown headers for logical grouping.
  - **Semantic:** Uses AI embeddings and cosine distance to detect topic changes and split chunks logically.
  - **Recursive:** Employs layout-aware recursive splitting to preserve sentence and table row integrity.
- **Granular CSV Indexing:** Every row in a CSV is indexed as a self-contained document to ensure high accuracy for tabular data (e.g., specific coverage or premium lists).
- **Structured Lookup:** Specialized routing for benefits CSVs with reduced context injection for high-precision queries.

### 🛠️ Admin Capabilities
- **Document Management:** Upload, re-ingest, or delete insurance policies through a dedicated UI.
- **Vector Automation:** Automatic text extraction, chunking, and embedding using AWS Bedrock and Pinecone.
- **User Control:** Manage user roles (Admin/User) and account status.

---

## 🏗️ Technical Stack

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **AI Orchestration:** [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
- **Large Language Model:** [AWS Bedrock](https://aws.amazon.com/bedrock/) (Claude 3.5 Sonnet / Titan Embeddings)
- **Vector Database:** [Pinecone](https://www.pinecone.io/)
- **Database:** SQLite (for session, user management, and AI conversation checkpoints)
- **Frontend:** Vanilla JS / HTML / CSS (Fast and lightweight)

---

## 📂 Project Structure

```text
├── app/
│   ├── main.py          # API Routes, Authentication & Admin Logic
│   ├── rag_service.py   # Advanced PDF/CSV Processing & AI Logic (LangChain/LangGraph)
│   ├── db.py            # SQLite Database & Directory Helpers
├── scripts/             # System validation, benchmarks, and test scripts
├── public/              # Frontend Assets (HTML/JS/CSS)
├── data/                # Local DB and checkpoint storage
└── uploads/             # Document upload storage
```

---

## ⚙️ Setup & Configuration

### Prerequisites
- Python 3.10+
- AWS Account (with Bedrock access)
- Pinecone API Key

### 1. Environment Variables
Create a `.env` file from the example:
```bash
cp .env.example .env
```

| Variable | Description |
| :--- | :--- |
| `SESSION_SECRET` | Secret key for signing session cookies. |
| `AUTH_USERNAME` | Bootstrap admin username. |
| `AUTH_PASSWORD` | Bootstrap admin password. |
| `AWS_*` | Access keys and region for Bedrock. |
| `PINECONE_*` | API key and Index details for vector storage. |

### 2. Local Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Docker Installation
```bash
docker compose up --build
```
Access the application at `http://localhost:8000`.

---

## 🔄 Document Workflow

1. **Ingestion:** Admin uploads a PDF/CSV -> System applies structural (Markdown) or row-based parsing -> Bedrock generates embeddings.
2. **Storage:** Chunks and vectors are stored in **Pinecone** under a unique namespace.
3. **Retrieval:** User asks a question -> System searches Pinecone for relevant context using semantic similarity.
4. **Generation:** Bedrock (Claude) synthesizes a direct answer based **strictly** on the retrieved context via a LangGraph-orchestrated agent.

---
*Developed for MediLink Insurance.*
