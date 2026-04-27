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

### 🛠️ Intelligent Document Parsing & Retrieval
- **Triple-Tier Hybrid PDF Parsing:**
  - **Structural (Markdownification):** Converts PDF layout to Markdown headers for logical grouping.
  - **Semantic:** Uses AI embeddings and cosine distance to detect topic changes and split chunks logically.
  - **Recursive:** Employs layout-aware recursive splitting to preserve sentence and table row integrity.
- **Advanced CSV Ingestion:**
  - **Metadata Carry-Forward:** Automatically fills merged or empty cells by propagating category headers (e.g., "Surgical Benefits") down to every sub-row.
  - **Pairwise Transformation:** Converts rows into semantically rich strings (`Header: Value;`) for superior embedding quality.
  - **Metadata Trimming:** Professional-grade lean storage that keeps only essential search and filtering keys.
- **Enterprise Retrieval Strategy:**
  - **Plan-Aware Hard Filtering:** Detects user plan mentions (e.g., "1.5M") and uses metadata to physically filter out irrelevant data at the database level.
  - **Hybrid Search (Semantic + Keyword):** Combines vector similarity with an weighted keyword scoring system to handle acronyms and technical terms (IP, ER, FNOL).
  - **Contextual Grouping (Parent-Child):** Automatically re-assembles fragmented table rows into complete "Benefit Families" to ensure the LLM sees the full context of a category.

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
uvicorn app.main:app --reload --port 8001
```

### 3. Docker Installation
```bash
docker compose up --build
```
Access the application at `http://localhost:8001`.

---

## 🔄 Document Workflow

1. **Ingestion:** Admin uploads a PDF/CSV -> Parser applies **Carry-Forward** logic for tables -> Rows are converted to **Pairwise Strings** -> Lean metadata is generated.
2. **Storage:** Vectors and **Filtered-Ready Metadata** are stored in **Pinecone**.
3. **Retrieval:** User asks a question -> System performs **Hybrid Search** -> Detects Plan constraints for **Hard Filtering** -> Applies **Contextual Grouping** to expand fragments into full benefit lists.
4. **Generation:** Bedrock (Claude) synthesizes a direct answer based **strictly** on the retrieved context. **Strict Guardrails** prevent medical inference or category guessing.

---
*Developed for MediLink Insurance.*
