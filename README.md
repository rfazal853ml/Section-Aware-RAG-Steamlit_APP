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

## 📋 Prerequisites

Ensure you have the following API keys:
1.  **Mistral AI API Key**
2.  **Pinecone API Key**

### Pinecone Web Setup
1.  Log in to your [Pinecone Dashboard](https://app.pinecone.io/).
2.  Create a **New Index**.
3.  **Index Name:** `unstructdocwithmetadata` (must match the name in `app.py`).
4.  **Dimensions:** Set to `1024` (to match `mistral-embed`).
5.  **Metric:** `cosine`.


---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -3.11 -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   Mistral_api_key=your_mistral_api_key_here
   Pinecone_api_key=your_pinecone_api_key_here
   ```

## 🏃 How to Run

Launch the application using Streamlit:
```bash
streamlit run app.py
```

---

## 📂 Project Structure
* `app.py`: The Streamlit frontend and chat session logic.
* `functionality.py`: The `RAGSystem` class orchestrating retrieval, scoring, and generation.
* `prompts.py`: Logic for the scoring rubric and the research assistant's persona.
* `requirements.txt`: Project dependencies.
* `questions.txt`: Example queries for testing the system.

---
*Developed as a precision-oriented information retriever for complex document structures.*