"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, then a
cross-encoder rerank.

This module is the reusable core of the project — it is import-clean (nothing
heavy runs at import; models and indexes load lazily on the first ``retrieve``
call) so other projects can ``from src.retrieve import HybridRetriever`` without
side effects.
"""
from dataclasses import dataclass

from .config import settings, resolve_device


@dataclass
class Passage:
    text: str
    body: str
    source: str
    title: str
    section: str
    score: float                 # final reranker score (or fused score if no rerank)
    dense_score: float = None
    bm25_score: float = None
    rerank_score: float = None


def reciprocal_rank_fusion(result_lists, rrf_k):
    """Fuse ranked lists of (payload, score) into one fused ranking.

    Returns {chunk_id: (payload, fused_score)} so the caller can sort.
    """
    fused = {}
    for results in result_lists:
        for rank, (payload, _score) in enumerate(results):
            key = payload["id"]
            if key not in fused:
                fused[key] = [payload, 0.0]
            fused[key][1] += 1.0 / (rrf_k + rank)
    return fused


class HybridRetriever:
    def __init__(self, dense_k=None, bm25_k=None, rrf_k=None, rerank_top_n=None,
                 embed_model=None, rerank_model=None, backend=None, rerank=True):
        self.dense_k = dense_k or settings.dense_k
        self.bm25_k = bm25_k or settings.bm25_k
        self.rrf_k = rrf_k or settings.rrf_k
        self.rerank_top_n = rerank_top_n or settings.rerank_top_n
        self.embed_model = embed_model or settings.embed_model
        self.rerank_model = rerank_model or settings.rerank_model
        self.backend = backend or settings.backend
        self.use_rerank = rerank
        self._embedder = None
        self._store = None
        self._bm25 = None
        self._reranker = None

    # -- lazy components ------------------------------------------------------
    def _ensure_loaded(self):
        if self._embedder is None:
            from .embed import Embedder
            from .store import Bm25Index, DenseStore
            self._embedder = Embedder(self.embed_model)
            self._store = DenseStore(self.backend)
            self._bm25 = Bm25Index.load()

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            device = resolve_device(settings.rerank_device)
            self._reranker = CrossEncoder(self.rerank_model, device=device)
            if device == "cuda":
                self._reranker.model.half()
        return self._reranker

    # -- retrieval ------------------------------------------------------------
    def retrieve(self, query, top_n=None):
        self._ensure_loaded()
        top_n = top_n or self.rerank_top_n

        qv = self._embedder.encode(query)
        dense = self._store.search(qv, self.dense_k)
        sparse = self._bm25.search(query, self.bm25_k)

        dense_by_id = {p["id"]: s for p, s in dense}
        bm25_by_id = {p["id"]: s for p, s in sparse}

        fused = reciprocal_rank_fusion([dense, sparse], self.rrf_k)
        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)

        if self.use_rerank:
            candidates = ranked  # small corpus: rerank the full fused pool
            scores = self.reranker.predict(
                [[query, p["text"]] for p, _ in candidates])
            order = sorted(range(len(candidates)),
                           key=lambda i: scores[i], reverse=True)[:top_n]
            chosen = [(candidates[i][0], float(scores[i])) for i in order]
        else:
            chosen = [(p, fs) for p, fs in ranked[:top_n]]

        passages = []
        for payload, final in chosen:
            passages.append(Passage(
                text=payload["text"],
                body=payload.get("body", payload["text"]),
                source=payload["source"],
                title=payload.get("title", payload["source"]),
                section=payload.get("section", ""),
                score=final,
                dense_score=dense_by_id.get(payload["id"]),
                bm25_score=bm25_by_id.get(payload["id"]),
                rerank_score=final if self.use_rerank else None,
            ))
        return passages
