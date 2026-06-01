"""FastAPI service exposing the RAG pipeline.

    uvicorn app.api:app --host 0.0.0.0 --port 8000

The pipeline (and its models) load lazily on the first /chat call so the server
starts instantly; hit /health to check readiness.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import RagPipeline

app = FastAPI(title="CS2 RAG Assistant")
_pipeline = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


class ChatRequest(BaseModel):
    question: str
    top_n: int | None = None


class Source(BaseModel):
    n: int
    source: str
    section: str
    title: str
    body: str
    rerank_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "ready": _pipeline is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = pipeline().answer(req.question, top_n=req.top_n)
    sources = [
        Source(
            n=i, source=p.source, section=p.section, title=p.title, body=p.body,
            rerank_score=p.rerank_score, dense_score=p.dense_score,
            bm25_score=p.bm25_score,
        )
        for i, p in enumerate(result.passages, 1)
    ]
    return ChatResponse(answer=result.answer, sources=sources)
