# AI & RAG Exploration Repository

This repository is a collection of Jupyter Notebooks dedicated to learning, experimenting, and implementing various **Retrieval-Augmented Generation (RAG)** techniques. It explores different paradigms of providing context to Large Language Models (LLMs), primarily running locally via Ollama.

The primary dataset used across these experiments is the seminal paper *"Attention Is All You Need"* (`NIPS-2017-attention-is-all-you-need-Paper.pdf`).

## 📂 Repository Contents

### 1. `RAG.ipynb`
A foundational notebook demonstrating the basic concepts of RAG. It covers how to provide custom context to an LLM so it can answer questions based on specific knowledge rather than just its pre-trained weights.

### 2. `embedding.ipynb`
An exploration into **text embeddings**. This notebook demonstrates how to convert text into high-dimensional vector representations using embedding models. Understanding this is a crucial prerequisite for standard Vector RAG.

### 3. `RAG_pdf.ipynb` (Traditional Vector RAG)
A complete pipeline implementing traditional **Vector-based RAG** over a PDF document.
- **Workflow:** Loads the PDF → Chunks the text (`RecursiveCharacterTextSplitter`) → Generates embeddings (`SentenceTransformer`) → Stores them in a vector database (`ChromaDB`) → Retrieves the top K semantically similar chunks based on user query → Generates answers using `llama3.2:3b`.
- **Strengths:** Excellent for broad, semantic search and similarity matching.

### 4. `Graph_RAG.ipynb` (Vector-less Graph RAG)
An alternative approach to RAG that completely abandons embeddings and vector databases in favor of a **Knowledge Graph**.
- **Workflow:** Loads text and chunks it → Uses the LLM to extract explicit facts as `(Subject, Relation, Object)` triplets → Builds a directed graph using `networkx` → Extracts entities from the user's query → Finds exact node matches in the graph → Retrieves the 1-hop subgraph (neighboring facts) → Generates an answer.
- **Strengths:** High precision, 100% explainability, and excellent at multi-hop reasoning (e.g., A is connected to B, and B is connected to C).

## 🛠️ Technology Stack
All notebooks are built to run locally and privately.
- **Local LLM:** [Ollama](https://ollama.com/) running the `llama3.2:3b` model.
- **Orchestration:** `langchain` and `langchain_ollama`
- **Vector Database:** `chromadb`
- **Graph Database:** `networkx` (In-memory)
- **Embeddings:** `SentenceTransformer` (`all-MiniLM-L6-v2`)

## 🚀 Getting Started

1. Ensure you have Python installed and create a virtual environment (`.venv`).
2. Install the required dependencies:
   ```bash
   pip install langchain langchain-community langchain-ollama chromadb sentence-transformers pypdf networkx matplotlib jupyter
   ```
3. Install [Ollama](https://ollama.com/) and download the required model:
   ```bash
   ollama run llama3.2:3b
   ```
4. Start a Jupyter server and open any of the notebooks!
