import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

DATASET_PATH = os.getenv("DATASET_PATH", "./GenAI Takeaway Assignment/data/jobs.csv")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vectorstore/index.faiss")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./vectorstore/docstore.jsonl")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

COHERE_API_KEY = "Your API key here"