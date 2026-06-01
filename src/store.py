"""Persistence for the two retrieval indexes.

* ``DenseStore`` wraps Qdrant (local on-disk by default, or a server when
  ``QDRANT_URL`` is set) with Chroma as a zero-infra fallback.
* ``Bm25Index`` is a small pickled lexical index built with ``rank_bm25``.

Both return the same shape: a list of ``(payload, score)`` where ``payload`` is
the chunk dict produced by :mod:`src.ingest`.
"""
import pickle
import re
from pathlib import Path

import numpy as np

from .config import settings

_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _token_re.findall(text.lower())


class DenseStore:
    def __init__(self, backend=None, dim=None):
        self.backend = backend or settings.backend
        self.dim = dim
        self._client = None
        self._collection = None  # chroma collection handle

    # -- qdrant ---------------------------------------------------------------
    def _qdrant(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            if settings.qdrant_url:
                self._client = QdrantClient(url=settings.qdrant_url)
            else:
                Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=settings.qdrant_path)
        return self._client

    # -- chroma ---------------------------------------------------------------
    def _chroma(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=settings.chroma_path)
            self._collection = client.get_or_create_collection(
                settings.collection, metadata={"hnsw:space": "cosine"})
        return self._collection

    def recreate(self, dim):
        self.dim = dim
        if self.backend == "qdrant":
            from qdrant_client.models import Distance, VectorParams
            self._qdrant().recreate_collection(
                settings.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            import chromadb
            client = chromadb.PersistentClient(path=settings.chroma_path)
            try:
                client.delete_collection(settings.collection)
            except Exception:
                pass
            self._collection = client.get_or_create_collection(
                settings.collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, vectors, payloads):
        if self.backend == "qdrant":
            from qdrant_client.models import PointStruct
            points = [PointStruct(id=i, vector=v.tolist(), payload=p)
                      for i, (v, p) in enumerate(zip(vectors, payloads))]
            self._qdrant().upsert(settings.collection, points=points)
        else:
            col = self._chroma()
            col.add(
                ids=[str(i) for i in range(len(payloads))],
                embeddings=[v.tolist() for v in vectors],
                documents=[p["text"] for p in payloads],
                metadatas=[{k: v for k, v in p.items() if k != "text"}
                           for p in payloads],
            )

    def search(self, vector, k):
        if self.backend == "qdrant":
            hits = self._qdrant().query_points(
                settings.collection, query=vector.tolist(), limit=k,
                with_payload=True).points
            return [(h.payload, float(h.score)) for h in hits]
        col = self._chroma()
        res = col.query(query_embeddings=[vector.tolist()], n_results=k,
                        include=["metadatas", "documents", "distances"])
        out = []
        for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0],
                                   res["distances"][0]):
            payload = dict(meta); payload["text"] = doc
            out.append((payload, 1.0 - float(dist)))  # cosine distance -> sim
        return out


class Bm25Index:
    def __init__(self, bm25=None, payloads=None):
        self.bm25 = bm25
        self.payloads = payloads or []

    @classmethod
    def build(cls, payloads):
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([tokenize(p["text"]) for p in payloads])
        return cls(bm25, list(payloads))

    def save(self, path=None):
        path = Path(path or settings.bm25_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "payloads": self.payloads}, f)

    @classmethod
    def load(cls, path=None):
        with open(path or settings.bm25_path, "rb") as f:
            d = pickle.load(f)
        return cls(d["bm25"], d["payloads"])

    def search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:k]
        return [(self.payloads[i], float(scores[i])) for i in idx]
