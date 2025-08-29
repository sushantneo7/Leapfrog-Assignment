from rank_bm25 import BM25Okapi
from typing import List
import re

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())

class BM25Index:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.tokens = [ _tokenize(t) for t in texts ]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, top_k: int) -> List[float]:
        qtokens = _tokenize(query)
        scores = self.bm25.get_scores(qtokens)
        return scores
