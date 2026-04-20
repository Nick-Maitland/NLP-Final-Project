from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq import dense_validation as dense_validation_module
from ragfaq import embeddings as embedding_module
from ragfaq import vector_store as vector_store_module
from ragfaq.config import ensure_runtime_directories, get_paths


class _FakeVector(list):
    def tolist(self) -> list[float]:
        return list(self)


def _prepare_dense_root(tmp_path: Path) -> Path:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "doc_a.md").write_text(
        "# Doc A\n\nSelf-attention compares tokens in the same sequence.",
        encoding="utf-8",
    )
    (knowledge_base / "doc_b.md").write_text(
        "# Doc B\n\nTransformers can process tokens in parallel within a layer.",
        encoding="utf-8",
    )
    (knowledge_base / "doc_c.md").write_text(
        "# Doc C\n\nMetadata keeps retrieved chunks traceable to their source.",
        encoding="utf-8",
    )
    (tmp_path / "PROJECT_REPORT.md").write_text(
        "\n".join(
            [
                "# Project Report",
                "",
                dense_validation_module.DENSE_VALIDATION_START,
                "placeholder",
                dense_validation_module.DENSE_VALIDATION_END,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_dense_validation_skips_when_chromadb_import_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _prepare_dense_root(tmp_path)
    monkeypatch.setenv("RAGFAQ_ROOT", str(root))
    paths = ensure_runtime_directories(get_paths())

    def _raise_import_error():
        raise ImportError("shadowed chromadb import")

    monkeypatch.setattr(vector_store_module, "_load_chromadb_module", _raise_import_error)

    summary = dense_validation_module.run_dense_validation(paths)
    assert summary["status"] == "skipped"
    assert "chromadb import unavailable" in summary["reason"]
    assert summary["checks"][0]["status"] == "skipped"
    assert summary["checks"][1]["status"] == "not_run"


def test_dense_validation_skips_when_sentence_transformers_import_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _prepare_dense_root(tmp_path)
    monkeypatch.setenv("RAGFAQ_ROOT", str(root))
    paths = ensure_runtime_directories(get_paths())

    monkeypatch.setattr(vector_store_module, "_load_chromadb_module", lambda: object())

    def _raise_import_error():
        raise ImportError("shadowed sentence_transformers import")

    monkeypatch.setattr(embedding_module, "_load_sentence_transformer_class", _raise_import_error)

    summary = dense_validation_module.run_dense_validation(paths)
    assert summary["status"] == "skipped"
    assert "sentence_transformers import unavailable" in summary["reason"]
    assert summary["checks"][0]["status"] == "passed"
    assert summary["checks"][1]["status"] == "skipped"


def test_dense_validation_skips_when_minilm_cache_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _prepare_dense_root(tmp_path)
    monkeypatch.setenv("RAGFAQ_ROOT", str(root))
    paths = ensure_runtime_directories(get_paths())

    monkeypatch.setattr(vector_store_module, "_load_chromadb_module", lambda: object())
    monkeypatch.setattr(embedding_module, "_load_sentence_transformer_class", lambda: object)
    monkeypatch.setattr(
        embedding_module.SentenceTransformerEmbeddingProvider,
        "local_model_status",
        lambda self: (False, "model cache missing"),
    )

    summary = dense_validation_module.run_dense_validation(paths)
    assert summary["status"] == "skipped"
    assert "MiniLM local cache unavailable" in summary["reason"]
    assert summary["checks"][2]["status"] == "skipped"
    assert summary["checks"][3]["status"] == "not_run"


def test_dense_validation_passes_with_mocked_dense_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _prepare_dense_root(tmp_path)
    monkeypatch.setenv("RAGFAQ_ROOT", str(root))
    paths = ensure_runtime_directories(get_paths())

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs) -> None:
            self.model_name = model_name
            self.kwargs = kwargs

        def encode(self, texts, **kwargs):
            return [_FakeVector([float(index + 1), 0.0]) for index, _ in enumerate(texts)]

    class FakeCollection:
        def __init__(self) -> None:
            self.documents: list[str] = []
            self.metadatas: list[dict[str, str]] = []
            self.ids: list[str] = []

        def add(self, **kwargs) -> None:
            self.documents = list(kwargs["documents"])
            self.metadatas = list(kwargs["metadatas"])
            self.ids = list(kwargs["ids"])

        def delete(self, ids=None) -> None:
            if ids is None:
                return
            remaining = [
                item
                for item in zip(self.ids, self.documents, self.metadatas)
                if item[0] not in set(ids)
            ]
            self.ids = [item[0] for item in remaining]
            self.documents = [item[1] for item in remaining]
            self.metadatas = [item[2] for item in remaining]

        def count(self) -> int:
            return len(self.ids)

        def query(self, **kwargs):
            count = kwargs["n_results"]
            distances = [round(0.05 * (index + 1), 2) for index in range(count)]
            return {
                "documents": [self.documents[:count]],
                "metadatas": [self.metadatas[:count]],
                "distances": [distances],
            }

    class FakeClient:
        def __init__(self) -> None:
            self.collections: dict[str, FakeCollection] = {}

        def get_or_create_collection(self, name: str):
            return self.collections.setdefault(name, FakeCollection())

        def get_collection(self, name: str):
            return self.collections[name]

        def delete_collection(self, name: str) -> None:
            self.collections.pop(name, None)

    fake_client = FakeClient()

    class FakeChromaModule:
        @staticmethod
        def PersistentClient(path: str):
            return fake_client

    monkeypatch.setattr(vector_store_module, "_load_chromadb_module", lambda: FakeChromaModule)
    monkeypatch.setattr(
        embedding_module,
        "_load_sentence_transformer_class",
        lambda: FakeSentenceTransformer,
    )
    monkeypatch.setattr(
        embedding_module.SentenceTransformerEmbeddingProvider,
        "local_model_status",
        lambda self: (True, str(tmp_path / "cache")),
    )

    summary = dense_validation_module.run_dense_validation(paths, collection_name="validation_test")
    assert summary["status"] == "passed"
    assert summary["chunk_count"] == 3
    assert summary["stored_count"] == 3
    assert summary["retrieved_count"] == 3
    assert all(check["status"] == "passed" for check in summary["checks"])
    assert summary["retrieved_chunks"][0]["source_id"] == "doc_a"
    assert summary["retrieved_chunks"][0]["source"].endswith("knowledge_base/doc_a.md")


def test_dense_validation_artifacts_sync_project_report(tmp_path: Path, monkeypatch) -> None:
    root = _prepare_dense_root(tmp_path)
    monkeypatch.setenv("RAGFAQ_ROOT", str(root))
    paths = ensure_runtime_directories(get_paths())
    summary = {
        "status": "failed",
        "generated_at": "2026-04-20T00:00:00+00:00",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "collection_name": "ragfaq_dense_validation",
        "question": "What is self-attention?",
        "top_k": 3,
        "reason": "Expected exactly 3 retrieved chunks, but received 2.",
        "chunk_count": 3,
        "stored_count": 3,
        "retrieved_count": 2,
        "retrieved_chunks": [],
        "checks": [
            {
                "id": "top_3_results",
                "name": "Verify exactly top-3 chunks are returned",
                "status": "failed",
                "detail": "Expected exactly 3 retrieved chunks, but received 2.",
            }
        ],
    }

    dense_validation_module.write_dense_validation_artifacts(summary, paths=paths)

    report_text = paths.dense_validation_report_path.read_text(encoding="utf-8")
    project_report_text = paths.project_report_path.read_text(encoding="utf-8")
    assert "The dense path did not validate successfully in this environment." in report_text
    assert "### Implemented Dense Path" in project_report_text
    assert "### Validated Dense Path" in project_report_text
    assert "Latest status: `failed`" in project_report_text
    assert "Reason: Expected exactly 3 retrieved chunks, but received 2." in project_report_text


def test_dense_validation_renderers_distinguish_passed_skipped_and_failed() -> None:
    passed_payload = {
        "status": "passed",
        "generated_at": "2026-04-20T00:00:00+00:00",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "collection_name": "ragfaq_dense_validation",
        "question": "What is self-attention?",
        "top_k": 3,
        "reason": "",
        "chunk_count": 3,
        "stored_count": 3,
        "retrieved_count": 3,
        "retrieved_chunks": [],
        "checks": [],
    }
    skipped_payload = {**passed_payload, "status": "skipped", "reason": "chromadb import unavailable"}
    failed_payload = {**passed_payload, "status": "failed", "reason": "Stored chunk count mismatch"}

    passed_text = dense_validation_module.render_dense_validation_section(passed_payload)
    skipped_text = dense_validation_module.render_dense_validation_section(skipped_payload)
    failed_text = dense_validation_module.render_dense_validation_report(failed_payload)

    assert "The dense path was validated successfully in this environment." in passed_text
    assert "it was not validated successfully in this environment." in skipped_text
    assert "did not validate successfully in this environment." in failed_text
