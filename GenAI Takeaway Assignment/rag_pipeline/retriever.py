import numpy as np
from typing import List, Dict, Optional
from .vectorstore import FaissStore
from .hybrid_search import BM25Index
from .embeddings import embed_query
from .config import HYBRID_ALPHA

def _normalize(arr):
    arr = np.array(arr, dtype="float32")
    if arr.size == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

class Retriever:
    def __init__(self, store: FaissStore, bm25: BM25Index):
        self.store = store
        self.bm25 = bm25

    def retrieve(self, query: str, top_k: int, filters: Optional[Dict] = None):
        qvec = embed_query(query)
        faiss_hits = self.store.search(qvec, top_k=top_k*5)  # over-fetch
        bm25_scores = self.bm25.search(query, top_k=top_k*5)

        N = len(self.store.texts)
        bm25_arr = np.zeros(N, dtype="float32")
        bm25_arr[:len(bm25_scores)] = bm25_scores

        faiss_idxs = [i for i,_ in faiss_hits]
        bm25_top_idxs = list(np.argsort(bm25_arr)[::-1][:top_k*5])
        candidate_idxs = list(dict.fromkeys(faiss_idxs + bm25_top_idxs))

        faiss_score_arr = np.zeros(N, dtype="float32")
        for i, s in faiss_hits:
            faiss_score_arr[i] = s
        faiss_norm = _normalize(faiss_score_arr)
        bm25_norm = _normalize(bm25_arr)
        combo = HYBRID_ALPHA * bm25_norm + (1 - HYBRID_ALPHA) * faiss_norm

        scored = [(idx, float(combo[idx])) for idx in candidate_idxs]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored:
            md = self.store.metadatas[idx]
            if filters:
                ok = True
                for k,v in filters.items():
                    mv = (md.get(k) or "").lower()
                    if str(v).lower() not in mv:
                        ok = False; break
                if not ok: 
                    continue
            results.append({"text": self.store.texts[idx], "metadata": md, "score": score})
            if len(results) >= top_k:
                break
        return results