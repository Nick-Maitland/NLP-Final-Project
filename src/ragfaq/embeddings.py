from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from .config import DENSE_MODEL_NAME, EMBED_BATCH_SIZE, EMBED_DEVICE
from .utils import RagFaqError


def _load_sentence_transformer_class():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def _model_cache_candidates(cache_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        candidates.append(Path(env_hub).expanduser())

    env_transformers = os.environ.get("TRANSFORMERS_CACHE")
    if env_transformers:
        candidates.append(Path(env_transformers).expanduser())

    env_hf_home = os.environ.get("HF_HOME")
    if env_hf_home:
        hf_home = Path(env_hf_home).expanduser()
        candidates.append(hf_home)
        candidates.append(hf_home / "hub")

    candidates.extend(
        [
            cache_dir / "hf_home",
            cache_dir / "hf_home" / "hub",
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "torch" / "sentence_transformers",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def find_local_model_cache(
    cache_dir: Path,
    model_name: str = DENSE_MODEL_NAME,
) -> tuple[bool, str]:
    hub_pattern = f"models--{model_name.replace('/', '--')}"
    st_pattern = model_name.replace("/", "_")
    for candidate in _model_cache_candidates(cache_dir):
        if not candidate.exists():
            continue
        if (candidate / hub_pattern).exists():
            return True, str(candidate / hub_pattern)
        if (candidate / st_pattern).exists():
            return True, str(candidate / st_pattern)
    return False, f"{model_name} is not cached in known local model locations."


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, cache_dir: Path, model_name: str = DENSE_MODEL_NAME) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self._model = None

    def _configure_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def local_model_status(self) -> tuple[bool, str]:
        return find_local_model_cache(self.cache_dir, self.model_name)

    def _load_model(self):
        self._configure_cache()
        if self._model is not None:
            return self._model

        try:
            sentence_transformer_cls = _load_sentence_transformer_class()
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise RagFaqError(
                "Dense retrieval is unavailable because sentence-transformers "
                f"could not be imported: {type(exc).__name__}: {exc}"
            ) from exc

        cached, detail = self.local_model_status()
        if not cached:
            raise RagFaqError(
                "Dense retrieval is unavailable because the MiniLM embedding model is "
                f"not available locally. {detail}"
            )

        try:
            self._model = sentence_transformer_cls(
                self.model_name,
                device=EMBED_DEVICE,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise RagFaqError(
                "Dense retrieval is unavailable because the MiniLM model could "
                f"not be loaded from the local cache: {type(exc).__name__}: {exc}"
            ) from exc
        return self._model

    def ensure_model_loaded(self):
        return self._load_model()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self.ensure_model_loaded()
        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
