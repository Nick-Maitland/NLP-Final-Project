from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.chunking import chunk_documents_with_report
from ragfaq.config import get_paths
from ragfaq.ingest import load_documents
from ragfaq.utils import RagFaqError


def _write_sample_faq_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "source_id,question,answer,topic,difficulty",
                (
                    "faq_attention_001,What is self-attention?,"
                    "Self-attention lets each token weigh other tokens in the same sequence.,"
                    "attention,intro"
                ),
                (
                    "faq_rag_001,What is retrieval-augmented generation?,"
                    "Retrieval-augmented generation combines retrieval with answer generation so "
                    "responses can use external evidence.,rag,intermediate"
                ),
            ]
        ),
        encoding="utf-8",
    )


def test_faq_csv_loads_with_metadata_and_stable_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    knowledge_base_dir = tmp_path / "knowledge_base"
    knowledge_base_dir.mkdir()
    _write_sample_faq_csv(knowledge_base_dir / "faqs.csv")
    (knowledge_base_dir / "notes.md").write_text(
        "\n".join(
            [
                "---",
                "title: Notes on Encoders",
                "topic: bert_style_encoders",
                "kind: doc",
                "---",
                "# Notes on Encoders",
                "",
                "Encoder models build contextual representations from both left and right context.",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RAGFAQ_ROOT", str(tmp_path))
    documents = load_documents(get_paths())
    assert len(documents) == 3

    faq_document = next(document for document in documents if document.source_id == "faq_attention_001")
    assert faq_document.metadata["source"] == "knowledge_base/faqs.csv"
    assert faq_document.metadata["topic"] == "attention"
    assert faq_document.metadata["difficulty"] == "intro"
    assert faq_document.metadata["kind"] == "faq"

    chunks_a, report_a = chunk_documents_with_report(documents)
    chunks_b, report_b = chunk_documents_with_report(documents)
    assert [chunk.chunk_id for chunk in chunks_a] == [chunk.chunk_id for chunk in chunks_b]
    assert report_a["top_chunk_ids"] == report_b["top_chunk_ids"]
    assert all(chunk.metadata["source_id"] == chunk.source_id for chunk in chunks_a)
    assert all("chunk_index" in chunk.metadata for chunk in chunks_a)
    assert all("topic" in chunk.metadata for chunk in chunks_a)


def test_faq_csv_missing_columns_raises_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base_dir = tmp_path / "knowledge_base"
    knowledge_base_dir.mkdir()
    (knowledge_base_dir / "faqs.csv").write_text(
        "\n".join(
            [
                "source_id,question,answer,topic",
                "faq_bad_001,What is a tokenizer?,A tokenizer splits text into units.,tokenization",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RAGFAQ_ROOT", str(tmp_path))
    with pytest.raises(RagFaqError, match="missing required columns"):
        load_documents(get_paths())


def test_repo_faq_csv_has_unique_nonempty_required_fields() -> None:
    faq_path = REPO_ROOT / "knowledge_base" / "faqs.csv"
    with faq_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(__import__("csv").DictReader(handle))

    assert len(rows) >= 100
    required = ["source_id", "question", "answer", "topic", "difficulty"]
    for row in rows:
        for field in required:
            assert row[field].strip() != ""

    counts = Counter(row["source_id"].strip() for row in rows)
    duplicates = [source_id for source_id, count in counts.items() if count > 1]
    assert duplicates == []
