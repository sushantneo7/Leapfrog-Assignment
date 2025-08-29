import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from dotenv import load_dotenv

from rag_pipeline.vectorstore import FaissStore
from rag_pipeline.hybrid_search import BM25Index
from rag_pipeline.retriever import Retriever
from rag_pipeline.llm import summarize
from rag_pipeline.utils import parse_query_to_filters
from rag_pipeline.preprocessing import load_and_prepare
from rag_pipeline.embeddings import embed_texts

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH", "./data/jobs.csv")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vectorstore/index.faiss")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./vectorstore/docstore.jsonl")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K_DEFAULT = int(os.getenv("TOP_K", 5))

app = FastAPI(
    title="RAG based Job Search",
)

class QueryRequest(BaseModel):
    query: str = Field(..., example="Type Job Title, Company or Location")

class JobResult(BaseModel):
    id: str
    title: str
    company: str
    location: str
    category: str
    level: str
    tags: str
    score: float

class QueryResponse(BaseModel):
    summary: str
    results: List[JobResult]

_store = FaissStore()
_bm25 = None
_retriever = None

def build_index():
    print("Building new FAISS index from dataset...")
    texts, metadatas = load_and_prepare(DATASET_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    vectors = embed_texts(texts)
    _store.build(vectors, metadatas, texts)
    _store.save(VECTOR_DB_PATH, DOCSTORE_PATH)
    print("Index built and saved.")

@app.on_event("startup")
def startup_event():
    global _store, _bm25, _retriever
    if not (os.path.exists(VECTOR_DB_PATH) and os.path.exists(DOCSTORE_PATH)):
        build_index()
    else:
        _store.load(VECTOR_DB_PATH, DOCSTORE_PATH)
    _bm25 = BM25Index(_store.texts)
    _retriever = Retriever(_store, _bm25)
    print("RAG components ready.")

@app.post("/api/query", response_model=QueryResponse, summary="Search jobs")
def api_query(req: QueryRequest):
    """Search job postings with natural language query + optional filters."""
    top_k = TOP_K_DEFAULT
    auto_filters = parse_query_to_filters(req.query)

    hits = _retriever.retrieve(req.query, top_k, filters=auto_filters)

    # Fallback if filters return nothing
    if not hits and auto_filters:
        hits = _retriever.retrieve(req.query, top_k, filters=None)

    results = []
    for h in hits:
        m = h["metadata"]
        results.append(JobResult(
            id=m.get("id"),
            title=m.get("title"),
            company=m.get("company"),
            location=m.get("location"),
            category=m.get("category"),
            level=m.get("level"),
            tags=m.get("tags"),
            score=round(float(h.get("score", 0.0)), 4),
        ))

    summary = summarize(req.query, hits)
    return QueryResponse(summary=summary, results=results)