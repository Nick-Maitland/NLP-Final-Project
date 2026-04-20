from __future__ import annotations

import os
from pathlib import Path

from .config import DENSE_MODEL_NAME, EMBED_BATCH_SIZE, EMBED_DEVICE
from .utils import RagFaqError


class MiniLMEmbedder:
    def __init__(self, cache_dir: Path, model_name: str = DENSE_MODEL_NAME) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self._model = None

    def _configure_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        hf_home = self.cache_dir / "hf_home"
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))

    def _load_model(self):
        self._configure_cache()
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - depends on runtime environment
                raise RagFaqError(
                    "Dense retrieval is unavailable because sentence-transformers "
                    f"could not be imported: {type(exc).__name__}: {exc}"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name, device=EMBED_DEVICE)
            except Exception as exc:  # pragma: no cover - depends on runtime environment
                raise RagFaqError(
                    "Dense retrieval is unavailable because the MiniLM model could "
                    f"not be loaded: {type(exc).__name__}: {exc}"
                ) from exc
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
