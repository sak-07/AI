# AI & RAG Exploration Repository

This repository is a collection of Jupyter Notebooks dedicated to learning, experimenting, and implementing various **Retrieval-Augmented Generation (RAG)** techniques. It explores different paradigms of providing context to Large Language Models (LLMs), primarily running locally via Ollama.

The primary dataset used across these experiments is the seminal paper *"Attention Is All You Need"* (`NIPS-2017-attention-is-all-you-need-Paper.pdf`).

## 📂 Repository Contents

### 1. `embedding.ipynb`
An exploration into **text embeddings**. This notebook demonstrates how to convert text into high-dimensional vector representations using embedding models. Understanding this is a crucial prerequisite for standard Vector RAG.

### 2. `RAG.ipynb` & `RAG_pdf.ipynb` (Traditional Vector RAG)
A complete pipeline implementing traditional **Vector-based RAG** over text and PDF documents.
- **Workflow:** Loads the PDF → Chunks the text (`RecursiveCharacterTextSplitter`) → Generates embeddings (`SentenceTransformer`) → Stores them in a vector database (`ChromaDB`) → Retrieves the top K semantically similar chunks based on user query → Generates answers using local LLMs.
- **Strengths:** Excellent for broad, semantic search and similarity matching.

### 3. `Hybrid_Search_RAG.ipynb` (Hybrid Search RAG)
Explores combining traditional keyword-based search with semantic vector search to improve retrieval accuracy. 

### 4. `Graph_RAG.ipynb` (Vector-less Graph RAG)
An alternative approach to RAG that completely abandons embeddings and vector databases in favor of a **Knowledge Graph**.
- **Workflow:** Loads text and chunks it → Uses the LLM to extract explicit facts as `(Subject, Relation, Object)` triplets → Builds a directed graph using `networkx` → Extracts entities from the user's query → Finds exact node matches in the graph → Retrieves the 1-hop subgraph (neighboring facts) → Generates an answer.
- **Strengths:** High precision, 100% explainability, and excellent at multi-hop reasoning (e.g., A is connected to B, and B is connected to C).

### 5. `Function_Calling.ipynb`
A foundation for Agentic workflows. This notebook demonstrates how to give an LLM access to external Python functions (tools), allowing it to take actions rather than just generating text.

### 6. `ReAct_Agent.ipynb`
Builds an AI Agent from scratch using the **ReAct (Reason + Act)** pattern. Demonstrates how to use a `while` loop to let an LLM think, execute a tool, observe the result, and decide when to output a final answer.

### 7. `Agentic_RAG_Tutorial.ipynb`
Introduces Agentic RAG routing using both **LangGraph** and **LlamaIndex Workflows**. The LLM acts as a router, deciding whether a user's query requires searching a vector database or performing a web search.

### 8. `Self_RAG.ipynb` (Self-Reflective RAG)
Implements an advanced Agentic RAG pipeline using **LangGraph**. It adds a "Grader" step that evaluates whether retrieved documents actually answer the user's question, and automatically rewrites the search query to try again if the retrieval was poor.

## 🛠️ Technology Stack
All notebooks are built to run locally and privately.
- **Local LLMs:** [Ollama](https://ollama.com/) running `llama3.2:3b` and `qwen2.5:7b-instruct` models.
- **Orchestration:** `langchain`, `langchain_ollama`, `langgraph`, and `llama-index`
- **Vector Database:** `chromadb`
- **Graph Database:** `networkx` (In-memory)
- **Embeddings:** `SentenceTransformer` (`all-MiniLM-L6-v2`)

## 🚀 Getting Started

1. Ensure you have Python installed and create a virtual environment (`.venv`).
2. Install the required dependencies:
   ```bash
   pip install langchain langchain-community langchain-ollama chromadb sentence-transformers pypdf networkx matplotlib jupyter langgraph llama-index-core llama-index-llms-ollama
   ```
3. Install [Ollama](https://ollama.com/) and download the required models:
   ```bash
   ollama run llama3.2:3b
   ollama run qwen2.5:7b-instruct
   ```
4. Start a Jupyter server and open any of the notebooks!
