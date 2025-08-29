import os, json
import numpy as np
import faiss
from typing import List, Dict, Tuple

class FaissStore:
    def __init__(self, dim: int = None):
        self.index = None
        self.dim = dim
        self.metadatas = []
        self.texts = []

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    def build(self, embeddings: List[List[float]], metadatas: List[Dict], texts: List[str]):
        vecs = np.array(embeddings, dtype="float32")
        vecs = self._normalize(vecs)
        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # cosine via normalized dot
        self.index.add(vecs)
        self.metadatas = metadatas
        self.texts = texts

    def save(self, index_path: str, docstore_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(docstore_path, "w") as f:
            for md, txt in zip(self.metadatas, self.texts):
                rec = {"text": txt, "metadata": md}
                f.write(json.dumps(rec) + "\n")

    def load(self, index_path: str, docstore_path: str):
        self.index = faiss.read_index(index_path)
        self.metadatas, self.texts = [], []
        with open(docstore_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                self.texts.append(rec["text"])
                self.metadatas.append(rec["metadata"])

    def search(self, query_vec: List[float], top_k: int) -> List[Tuple[int, float]]:
        import numpy as np
        q = np.array([query_vec], dtype="float32")
        q = self._normalize(q)
        D, I = self.index.search(q, top_k)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0])]