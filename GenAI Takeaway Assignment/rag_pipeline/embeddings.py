from typing import List
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True).tolist()

def embed_query(text: str) -> List[float]:
    model = _get_model()
    return model.encode([text], convert_to_numpy=True)[0].tolist()
