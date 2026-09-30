# 📚 DocuBase

**DocuBase** is an end-to-end, high-performance Document Processing and Retrieval-Augmented Generation (RAG) system built with Python. It ingests multi-format documents, generates dense vector embeddings using Sentence Transformers, stores them in ChromaDB, and performs semantic search and context-aware QA using OpenAI-compatible LLM endpoints (e.g., NVIDIA NIM).

---

## ✨ Features

- **Multi-Format Document Loading**: Built-in support for loading document formats recursively from directories:
  - PDF (`.pdf`) via PyMuPDF
  - Word Documents (`.docx`) via Docx2txt
  - Text Files (`.txt`) via TextLoader
  - CSV Spreadsheets (`.csv`) via CSVLoader
  - Excel Spreadsheets (`.xlsx`) via Unstructured Excel Loader
  - JSON Files (`.json`) via JSONLoader
- **Recursive Text Chunking**: Splits large documents into overlapping semantic chunks using `RecursiveCharacterTextSplitter` with customizable chunk sizes and overlaps.
- **Vector Embeddings**: Generates dense numerical embeddings using `SentenceTransformer` models (default: `all-MiniLM-L6-v2`).
- **Persistent Vector Store**: Powered by **ChromaDB** (`chromadb.PersistentClient`) for fast persistent storage, similarity search, and collection management.
- **RAG Search Pipeline**: Integrates with LLMs (NVIDIA NIM / OpenAI API via `langchain-openai`) to synthesize concise, context-bound answers with source citations (`[Source 1]`, `[Source 2]`, etc.).

---

## 📁 Project Architecture & Structure

```
DocuBase/
├── data/                       # Document storage directory
│   └── vector_store/           # Persistent ChromaDB vector database files
├── notebook/                   # Research & exploratory Jupyter notebooks
│   ├── document.ipynb
│   └── pdf_loader.ipynb
├── src/                        # Core Python package modules
│   ├── __init__.py             # Module exports & package metadata
│   ├── data_loader.py          # Multi-format document loader
│   ├── embeddings.py           # Text chunking & SentenceTransformer embeddings
│   ├── vector_store.py         # ChromaDB persistence & similarity search
│   └── search.py               # RAG search pipeline & LLM integration
├── .env                        # Environment variables (API keys & endpoints)
├── app.py                      # Main application execution script
├── main.py                     # Execution entry point
├── pyproject.toml              # Project dependencies & configuration
├── requirements.txt            # Dependency specification
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites

- **Python**: Version `3.13` or higher
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### 1. Clone & Setup Environment

Using `uv`:
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

Or using standard `pip`:
```bash
python -m venv .venv
.venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_NAME=nvidia/llama-3.1-nemotron-70b-instruct
```

---

## 🚀 Quickstart & Usage

### 1. Ingest Documents & Build Vector Store

Place your documents (PDF, DOCX, CSV, TXT, XLSX, JSON) inside the `data/` directory, then initialize and build the ChromaDB store:

```python
from src import load_all_docs, ChromadbStore

# 1. Load documents from data directory
docs = load_all_docs("data")

# 2. Build persistent ChromaDB store
store = ChromadbStore(persist_dir="data/vector_store")
store.build_from_documents(docs)
```

### 2. Semantic Vector Query

Perform direct vector similarity search against indexed document chunks:

```python
from src import ChromadbStore

store = ChromadbStore(persist_dir="data/vector_store")
store.load()

# Search top 5 relevant document chunks
results = store.query("What is a Decision Tree?", top_k=5)
for res in results:
    print(f"ID: {res['id']}, Distance: {res['distance']:.4f}")
    print(f"Content: {res['document'][:150]}...\n")
```

### 3. End-to-End RAG Answer Generation

Query the RAG pipeline to get synthesized answers based strictly on retrieved document context:

```python
from src import ChromadbStore, SearchPipeline

# Load vector store & search pipeline
store = ChromadbStore(persist_dir="data/vector_store")
store.load()

search = SearchPipeline()
answer = search.rag_simple(
    query="What is a Multilayer Perceptron?",
    rag_retriever=store,
    top_k=5
)

print(f"Answer:\n{answer}")
```

### 4. Running the Main Script

```bash
python app.py
```

---
## Changes in current version (v0.1.1)
### 1. Duplicate Content Prevention
- **Source Grouping & File Hashing**: In build_from_documents, documents are grouped by metadata["source"]. A SHA-256 hash is computed for the combined text content of each source file.
- **Deletion of Modified Chunks**: When a file is modified, its existing chunks are identified and deleted from Chroma before adding the newly split chunks. This ensures no old orphaned chunks remain in the vector store.
- **Incremental Skipping**: If a source file's content hash matches the stored hash in the collection, the file is skipped completely, saving compute and embedding model calls.
- **Purging Deleted Files**: Any source files present in Chroma that are no longer present in the incoming documents list are purged from the vector store.
- **Deterministic Chunk IDs & Upsert**: Replaced random uuid4() IDs with content-based SHA-256 hashes and used self.collection.upsert().

## 🔮 Future Scope & Roadmap

### Web Application Interface
- **Interactive UI**: Develop a web application (e.g., Streamlit / FastAPI + web interface) for seamless user interaction:
  - Drag-and-drop document upload and batch ingestion.
  - Interactive Q&A chat interface with expandable source citations.
  - Vector database management dashboard (view total vectors, inspect chunk metadata, clear index).

---

## Some test runs with outputs
1. Loading Documents in vector store and querying the RAG pipeline

```
Loaded: data\Machine_Learning_Unit_1.pdf
Loaded: data\Machine_Learning_Unit_2.pdf
Loaded: data\Machine_Learning_Unit_3.pdf
Loaded: data\Machine_Learning_Unit_4.pdf
Loaded: data\Machine_Learning_Unit_5.pdf

Total documents loaded: 164
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3865.06it/s]
Loaded embedding model: all-MiniLM-L6-v2
Using Chroma collection 'test_collection' with 0 documents
Building/updating vector store from 164 raw documents...
New file detected: 'data\Machine_Learning_Unit_1.pdf'
New file detected: 'data\Machine_Learning_Unit_2.pdf'
New file detected: 'data\Machine_Learning_Unit_3.pdf'
New file detected: 'data\Machine_Learning_Unit_4.pdf'
New file detected: 'data\Machine_Learning_Unit_5.pdf'
Loading weights: 100%|███████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 4666.28it/s]

Loaded model: all-MiniLM-L6-v2
Split 164 documents into 267 chunks

Generating embeddings for 267 texts.
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████| 9/9 [00:35<00:00,  3.99s/it]
Generated embeddings with shape: (267, 384)
[INFO] Upserted 267 vectors to Chroma collection.
Vector store update completed and saved to J:\data\vector_store
[INFO] Loaded 267 documents from Chroma.
Enter your query or type 'exit' to quit: what are the types of machine learning
[INFO] Querying vector store for: 'what are the types of machine learning'
Answer:
Based on the provided document context, there are mainly three types of machine learning [Source 1]:

1. **Supervised Learning**
2. **Unsupervised Learning**
3. **Reinforcement Learning**
Enter your query or type 'exit' to quit: what is decision tree and what type of machine learning it is
[INFO] Querying vector store for: 'what is decision tree and what type of machine learning it is'
Answer:
Based on the provided document context:

**What is a Decision Tree?**
Decision Trees are models widely used for **classification and regression tasks** due to their interpretability and simplicity [Source 1]. They are constructed by recursively splitting data based on attributes (e.g., using entropy or information gain) 
until all instances are classified or no attributes remain [Source 1, Source 4].

**What type of machine learning is it?**
The document lists three main types of machine learning: **Supervised Learning, Unsupervised Learning, and Reinforcement Learning** [Source 3]. Since Decision Trees are explicitly used for **classification and regression** [Source 1]—which are hallmark tasks of **Supervised Learning**—they fall under the Supervised Learning category.
Enter your query or type 'exit' to quit: exit
```

2. Trying to load duplicate documents and asking questions outside of the document

```
Loaded: data\Machine_Learning_Unit_1.pdf
Loaded: data\Machine_Learning_Unit_2.pdf
Loaded: data\Machine_Learning_Unit_3.pdf
Loaded: data\Machine_Learning_Unit_4.pdf
Loaded: data\Machine_Learning_Unit_5.pdf

Total documents loaded: 164
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 1713.16it/s]
Loaded embedding model: all-MiniLM-L6-v2
Using Chroma collection 'test_collection' with 267 documents
Building/updating vector store from 164 raw documents...
Skipping 'data\Machine_Learning_Unit_1.pdf' (unchanged, hash matches).
Skipping 'data\Machine_Learning_Unit_2.pdf' (unchanged, hash matches).
Skipping 'data\Machine_Learning_Unit_3.pdf' (unchanged, hash matches).
Skipping 'data\Machine_Learning_Unit_4.pdf' (unchanged, hash matches).
Skipping 'data\Machine_Learning_Unit_5.pdf' (unchanged, hash matches).
All documents are up-to-date. No modifications detected.
[INFO] Loaded 267 documents from Chroma.
Enter your query or type 'exit' to quit: Who is Albert
[INFO] Querying vector store for: 'Who is Albert'
Answer:
I do not know. The provided document context does not mention anyone named Albert.
Enter your query or type 'exit' to quit: exit
```

## 📜 License

This project is open-source under the MIT License.
