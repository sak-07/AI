# AI, LLM & RAG Exploration Repository

This repository is a comprehensive collection of Jupyter Notebooks dedicated to learning, experimenting, and implementing modern AI techniques. It covers everything from the fundamentals of Large Language Models (LLMs) and Hugging Face infrastructure, to advanced **Retrieval-Augmented Generation (RAG)** techniques, Agentic workflows, and Parameter-Efficient Fine-Tuning (PEFT). 

The primary dataset used across the RAG experiments is the seminal paper *"Attention Is All You Need"* (`NIPS-2017-attention-is-all-you-need-Paper.pdf`).

## 📂 Repository Contents

### 🧠 RAG & Vector Search
#### 1. `embedding.ipynb`
An exploration into **text embeddings**. This notebook demonstrates how to convert text into high-dimensional vector representations using embedding models. Understanding this is a crucial prerequisite for standard Vector RAG.

#### 2. `RAG.ipynb` & `RAG_pdf.ipynb` (Traditional Vector RAG)
A complete pipeline implementing traditional **Vector-based RAG** over text and PDF documents.
- **Workflow:** Loads the PDF → Chunks the text → Generates embeddings → Stores them in a vector database (`ChromaDB`) → Retrieves semantically similar chunks → Generates answers using local LLMs.
- **Strengths:** Excellent for broad, semantic search and similarity matching.

#### 3. `Hybrid_Search_RAG.ipynb` (Hybrid Search RAG)
Explores combining traditional keyword-based search with semantic vector search to improve retrieval accuracy. 

#### 4. `Graph_RAG.ipynb` (Vector-less Graph RAG)
An alternative approach to RAG that abandons embeddings in favor of a **Knowledge Graph**.
- **Workflow:** Uses the LLM to extract explicit facts as `(Subject, Relation, Object)` triplets → Builds a directed graph using `networkx` → Finds exact node matches in the graph → Retrieves the 1-hop subgraph.
- **Strengths:** High precision, 100% explainability, and excellent at multi-hop reasoning.

---

### 🤖 Agents & Advanced RAG
#### 5. `Function_Calling.ipynb`
A foundation for Agentic workflows. Demonstrates how to give an LLM access to external Python functions (tools), allowing it to take actions rather than just generating text.

#### 6. `ReAct_Agent.ipynb`
Builds an AI Agent from scratch using the **ReAct (Reason + Act)** pattern. Uses a `while` loop to let an LLM think, execute a tool, observe the result, and decide when to output a final answer.

#### 7. `Agentic_RAG_Tutorial.ipynb`
Introduces Agentic RAG routing using **LangGraph** and **LlamaIndex Workflows**. The LLM acts as a router, deciding whether a query requires searching a vector database or performing a web search.

#### 8. `Self_RAG.ipynb` (Self-Reflective RAG)
Implements an advanced pipeline using **LangGraph**. It adds a "Grader" step that evaluates whether retrieved documents actually answer the question, and automatically rewrites the search query if the retrieval was poor.

---

### 🏋️ Fine-Tuning & Evaluation
#### 9. `LoRA_PEFT_Tutorial.ipynb`
A tutorial demonstrating **Parameter-Efficient Fine-Tuning (PEFT)** using the **LoRA (Low-Rank Adaptation)** technique. We fine-tune an LLM to adopt a "sarcastic pirate" persona using the `SFTTrainer` and QLoRA for memory-efficient training. Output artifacts are stored in `lora_pirate_model/` and `pirate_lora_adapter/`.

#### 10. `RAGAS_Tutorial.ipynb`
A guide to evaluating RAG pipelines using the **RAGAS** framework. Demonstrates how to programmatically assess retrieval accuracy, generation quality, and answer relevance using the Gemini API.

---

### 📚 Fundamentals & Courses
#### 11. `Hands-On-Large-Language-Models/`
A structured walkthrough of core LLM concepts. Contains interactive notebooks covering:
- Chapter 1: Introduction to Language Models
- Chapter 2: Tokens and Token Embeddings
- Chapter 3: Looking Inside LLMs
- Chapter 4: Text Classification
- Chapter 5: Text Clustering and Topic Modeling

#### 12. `Huggingface/`
A hands-on exploration of the Hugging Face ecosystem, adapted from the official Hugging Face NLP Course.
- **1. Transformer Models**: High-level overview of the Transformer architecture.
- **2. Using Transformers**: Deep dives into the internals of the `pipeline`, Tokenizers, and PyTorch models.
- **3. Fine tuning**: Step-by-step guides on processing datasets efficiently with `Dataset.map()`, dynamic padding with data collators, and building full training loops using the `Trainer` API.

---

## 🛠️ Technology Stack
Many notebooks are built to run locally and privately.
- **Local LLMs:** [Ollama](https://ollama.com/) running `llama3.2:3b` and `qwen2.5:7b-instruct` models.
- **Orchestration:** `langchain`, `langgraph`, and `llama-index`
- **Hugging Face:** `transformers`, `datasets`, `evaluate`, `peft`, `trl`
- **Vector Database:** `chromadb`
- **Graph Database:** `networkx` (In-memory)
- **Embeddings:** `SentenceTransformer` (`all-MiniLM-L6-v2`)

## 🚀 Getting Started

1. Ensure you have Python installed and create a virtual environment (`.venv`).
2. Install the required dependencies:
   ```bash
   pip install langchain langchain-community langchain-ollama chromadb sentence-transformers pypdf networkx matplotlib jupyter langgraph llama-index-core llama-index-llms-ollama transformers datasets evaluate peft trl
   ```
3. Install [Ollama](https://ollama.com/) and download the required models:
   ```bash
   ollama run llama3.2:3b
   ollama run qwen2.5:7b-instruct
   ```
4. Start a Jupyter server and open any of the notebooks!
