# Evaluation Framework Documentation

This document explains how to test the conversational logic, retrieval quality, and system robustness of the SHL Assessment Recommender.

## Overview of Tests

The system uses `pytest` for all unit and integration testing. Tests are located in the `tests/` directory.

To run the entire test suite:
```bash
pytest tests/ -v
```

### 1. Engine & Logic Tests (`test_engines.py`, `test_clarification.py`)
These tests ensure that the core business logic behaves correctly without spinning up the FastAPI server or hitting vector stores unnecessarily.
- **Refusal Engine**: Validates that prompt injections, off-topic requests, and legal/salary queries are instantly rejected.
- **Comparison Engine**: Verifies that comparisons only use catalog-grounded data and don't hallucinate extra features.
- **Clarification Engine**: Checks the `sufficiency_score` logic. It ensures that if a user only provides a skill ("I need a Python test") without a role, the engine correctly requests clarification before searching.

### 2. State Reconstruction Tests (`test_conversation_state.py`)
Tests the `ConversationState.from_messages` method. It verifies that:
- The system can correctly extract roles, skills, and seniority from a multi-turn conversation.
- Contradictory information in later turns correctly updates the state.

### 3. Embedding & Retrieval Tests (`test_embedding_pipeline.py`)
Uses an ephemeral, in-memory ChromaDB instance to test retrieval logic without polluting your local `data/chroma_db` database.
- **Top-K Retrieval**: Verifies the vector store returns the expected number of results.
- **Similarity**: Ensures that related concepts (e.g., "Python" and "Java") yield closer cosine similarity scores than unrelated concepts.
- **Metadata Filtering**: Checks that duration and test type filters apply correctly during retrieval.

### 4. API Endpoint Tests (`test_chat_endpoint.py`, `test_health.py`)
Fires actual HTTP requests via `fastapi.testclient.TestClient`.
- Validates that the `/api/v1/chat` endpoint conforms EXACTLY to the specified output schema (`reply`, `recommendations`, `end_of_conversation`).

---

## Future Implementations: Recall@10 & Groundedness

Currently, the Hybrid Ranker utilizes hardcoded component weights:
- 0.45 Semantic (ChromaDB)
- 0.25 Keyword Overlap
- 0.20 Role Match
- 0.10 Seniority Match

To evaluate the mathematical accuracy of these weights in a production environment, a **Recall@10 Evaluation Script** must be implemented.

**How to implement a Recall@10 framework:**
1. Create a "golden dataset" (`golden_queries.json`) containing user queries and the exact URLs of the SHL assessments that *should* be recommended.
2. Build an `evaluate_recall.py` script that loops through the golden dataset, passes the query to `RetrievalService.search()`, and checks if the expected URL is present in the `top_k=10` results.
3. Calculate the aggregate percentage: `(Successful Hits) / (Total Queries) = Recall@10 Score`.
