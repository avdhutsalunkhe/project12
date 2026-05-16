# SHL Assessment Recommender

A conversational, stateless API backend that recommends SHL assessments based on natural-language queries. Built with FastAPI, Sentence-Transformers, ChromaDB, and a custom hybrid-ranking engine.

## 🚀 Features

- **Conversational Interface**: Chat with the system in natural language to find the best SHL assessments.
- **Hybrid Retrieval System**: Combines semantic search (ChromaDB + all-MiniLM-L6-v2) with TF-IDF keyword overlap and targeted metadata scoring (role, seniority).
- **Stateless Architecture**: Client manages conversation history. The server reconstructs intent on every turn.
- **Clarification Engine**: Detects vague queries (via dynamic sufficiency threshold) and asks targeted questions to prevent hallucinated guesses.
- **Zero Hallucination Guard**: All recommendations are strictly validated against the canonical `catalog.json` before being returned.
- **Refusal Engine**: Automatically rejects prompt injections, off-topic requests (salary, EEOC), and geographical queries.

## 🏗️ Architecture

```text
Request (ChatHistory) -> Refusal Guard -> State Reconstruction -> Clarification Check
                                                                       |
                                                                       v
[Semantic Search] + [TF-IDF Search] -> Deduplication & Hybrid Rank -> Hallucination Guard -> Response
```

## 🛠️ Setup & Installation

### 1. Requirements
- Python 3.10+
- pip

### 2. Install Dependencies
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and configure as needed:
```env
APP_ENV=development
PORT=8000
LOG_LEVEL=INFO
ALLOWED_ORIGINS=["http://localhost:3000"]
```

### 4. Initialize the Vector Database
Before starting the server, you must embed the SHL catalog into the ChromaDB vector store:
```bash
python -m pipeline.ingest
```

### 5. Start the Server
```bash
python run.py
```
Or use Uvicorn directly:
```bash
uvicorn app.main:app --reload
```

## 🌍 Deployment (Render)

This project includes a `render.yaml` Blueprint for easy deployment to Render.
1. Connect your repository to Render.
2. Select **Blueprints** > **New Blueprint Instance**.
3. Render will automatically run `pip install`, build the Chroma database (`python -m pipeline.ingest`), and launch Gunicorn with optimal worker scaling.

## 📂 Project Structure

- `app/` - FastAPI application (routes, schemas, core logic, engines)
- `data/` - Holds `catalog.json` and local ChromaDB persistence.
- `pipeline/` - Embedding logic and vector store ingestion.
- `scraper/` - Tooling to build `catalog.json` from SHL's website.
- `tests/` - Pytest suite for engines, endpoints, and retrieval accuracy.
- `docs/` - System documentation (API, Data, Evaluation).
