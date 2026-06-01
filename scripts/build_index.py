"""Build the dense (Qdrant) and lexical (BM25) indexes from the corpus.

Run once after editing the corpus:

    python -m scripts.build_index
"""
import time

from src.embed import Embedder
from src.ingest import load_chunks
from src.store import Bm25Index, DenseStore


def main():
    t0 = time.time()
    chunks = load_chunks()
    payloads = [c.as_payload() for c in chunks]
    print(f"loaded {len(chunks)} chunks")

    embedder = Embedder()
    vectors = embedder.encode([c.text for c in chunks])
    print(f"embedded with {embedder.model_name} ({embedder.dim}d) on {embedder.device}")

    store = DenseStore()
    store.recreate(embedder.dim)
    store.upsert(vectors, payloads)
    print(f"upserted into {store.backend}")

    Bm25Index.build(payloads).save()
    print(f"built bm25 index ({len(payloads)} docs)")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
