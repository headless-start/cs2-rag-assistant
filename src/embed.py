"""Embedding model wrapper (bge-m3 by default).

The model is loaded lazily on first use so importing this module is cheap and
the rest of the code can construct an Embedder without paying for it.
"""
import numpy as np

from .config import settings, resolve_device


class Embedder:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or settings.embed_model
        self.device = resolve_device(device or settings.embed_device)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.device == "cuda":   # fp16 halves VRAM, negligible recall loss
                self._model.half()
        return self._model

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=16):
        single = isinstance(texts, str)
        out = self.model.encode(
            [texts] if single else list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,      # cosine == dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        return out[0] if single else out
