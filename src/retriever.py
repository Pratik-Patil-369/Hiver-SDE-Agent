"""Retriever: cosine search over historical customer messages.

FAISS used when installed (IndexFlatIP); otherwise sklearn NearestNeighbors
(brute-force cosine) — identical API so results are comparable.
"""
import numpy as np
import pandas as pd


class Retriever:
    def __init__(self, embeddings: np.ndarray, records: pd.DataFrame):
        self.records = records.reset_index(drop=True)
        self.embeddings = np.asarray(embeddings).astype("float32")
        self.backend = "sklearn"
        self.index = None
        try:
            import faiss  # type: ignore
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.embeddings)
            self.backend = "faiss"
        except Exception:
            self.backend = "sklearn"

    def search(self, query_embedding: np.ndarray, k=5):
        q = np.asarray(query_embedding).astype("float32").reshape(1, -1)
        if self.backend == "faiss":
            scores, idxs = self.index.search(q, k)
            scores, idxs = scores[0], idxs[0]
        else:
            # cosine since embeddings are L2-normalized
            sims = (self.embeddings @ q[0])
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs]
        results = []
        for s, i in zip(scores, idxs):
            if int(i) < 0 or int(i) >= len(self.records):
                continue
            results.append({"score": float(s), "record": self.records.iloc[int(i)].to_dict()})
        return results
