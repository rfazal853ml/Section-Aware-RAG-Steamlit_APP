# RAG Research Assistant for Complex Documents

This repository contains a specialized Retrieval-Augmented Generation (RAG) system designed to handle **complex structured documents**. Unlike standard RAG pipelines, this system is **section-aware** and utilizes a **two-step retrieval process** to ensure high precision: it first filters relevant documents based on summaries and then performs a granular search within those specific files.

## 🚀 Key Features
* **Section-Aware Retrieval:** Tracks and displays metadata including document source, section names, and page numbers.
* **Two-Stage Pipeline:** 1.  **Macro-search:** An LLM scores document summaries to identify the most relevant files.
    2.  **Micro-search:** Retrieves specific page hits only from the filtered documents to reduce noise.
* **Smart Filtering:** Uses a custom scoring rubric (0-100) to ensure the assistant only reads documents that truly match the user query.
* **Streamlit UI:** A clean, interactive chat interface with real-time status updates and source citations.

## 🛠️ Tech Stack
* **LLM:** `mistral-large-latest`
* **Embeddings:** `mistral-embed`
* **Vector Database:** `Pinecone`
* **Framework:** `LangChain`
* **Interface:** `Streamlit`

---
