# RAG Based Job Search (Sushant Neopane)

The system is a hybrid Retrieval Augmented Generation (RAG) pipeline that returns job listings for user queries. The RAG pipeline consists of various components such as preprocessing, embeddings, vector store, retriever, hybrid search and LLM integration coupled with thoughtful prompt design for accurate and concise LLM output. It also consists of an API endpoint, implemented using FastAPI. The main endpoint is a POST request to /api/query, designed to handle incoming queries in JSON format.

The system first extracts key details like company, location, and job level from the user’s query using a simple regex parser. Then it uses hybrid search to find the best job matches by combining two methods: dense search using sentence-transformer embeddings with a FAISS index, and sparse search using BM25 for keyword matching. The results are combined based on an adjustable weight (alpha). If a Cohere API key is available, the system provides an AI-generated summary of the top results, otherwise, it uses a basic fallback template. Finally, the system delivers a short, easy-to-read summary along with relevant job IDs and metadata.


## Installation

Clone the repository

```bash
git clone <repo-url>
cd genai_assignment
```

Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate	(on Mac/Linux)
venv\Scripts\activate       (on Windows)
```
Install dependencies

```bash
pip install -r requirements.txt
```

Configure environment variables (.env)
```bash
EMBED_MODEL=all-MiniLM-L6-v2
DATASET_PATH=./data/jobs.csv
VECTOR_DB_PATH=./vectorstore/index.faiss
DOCSTORE_PATH=./vectorstore/docstore.jsonl
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=5
HYBRID_ALPHA=0.6
COHERE_API_KEY=
```
	
Run the API
```bash
uvicorn main:app --reload
```
Open in Browser
```bash
http://127.0.0.1:8000/docs
```