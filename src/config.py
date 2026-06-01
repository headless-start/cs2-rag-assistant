"""Central configuration, read from the environment with sensible defaults.

Everything that another project (or a docker container) might want to override
lives here so the rest of the code can stay free of os.environ lookups.
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass
class Settings:
    # data + index locations
    corpus_dir: Path = ROOT / "data" / "corpus"
    qdrant_path: str = str(ROOT / "qdrant_storage")   # local on-disk mode
    qdrant_url: str = _env("QDRANT_URL", "")           # set -> use a server instead
    collection: str = _env("QDRANT_COLLECTION", "cs2")
    bm25_path: str = str(ROOT / "qdrant_storage" / "bm25.pkl")
    backend: str = _env("VECTOR_BACKEND", "qdrant")    # qdrant | chroma
    chroma_path: str = str(ROOT / "chroma_storage")

    # models
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    gen_model: str = _env("GEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    # devices ("auto" -> cuda if available else cpu); set per-component so a
    # small machine can keep the generator on the GPU and push the
    # embedder/reranker onto the CPU.
    embed_device: str = _env("EMBED_DEVICE", "auto")
    rerank_device: str = _env("RERANK_DEVICE", "auto")
    gen_device: str = _env("GEN_DEVICE", "auto")

    # chunking
    chunk_tokens: int = int(_env("CHUNK_TOKENS", "280"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "50"))

    # retrieval
    dense_k: int = int(_env("DENSE_K", "20"))
    bm25_k: int = int(_env("BM25_K", "20"))
    rrf_k: int = int(_env("RRF_K", "60"))
    rerank_top_n: int = int(_env("RERANK_TOP_N", "5"))

    # generation provider: hf (local transformers) | openai (compatible endpoint)
    llm_provider: str = _env("LLM_PROVIDER", "hf")
    openai_base_url: str = _env("OPENAI_BASE_URL", "")
    openai_api_key: str = _env("OPENAI_API_KEY", "")
    max_new_tokens: int = int(_env("MAX_NEW_TOKENS", "512"))

    # api
    api_url: str = _env("API_URL", "http://localhost:8000")


settings = Settings()


def resolve_device(choice):
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
