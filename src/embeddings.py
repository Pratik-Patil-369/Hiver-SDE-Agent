"""Embeddings: TF-IDF by default (fast, offline). Sentence-transformers optional."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self, max_features=20000):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features)

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def encode(self, texts):
        import numpy as np
        m = self.vec.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        return m / norms


class SentenceEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self  # pretrained, no fit needed

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True).astype("float32")


def get_embedder(prefer="auto"):
    if prefer in ("auto", "sentence"):
        try:
            e = SentenceEmbedder()
            # lazy check: only succeed if package importable
            return e, "sentence"
        except Exception:
            pass
    return TfidfEmbedder(), "tfidf"
