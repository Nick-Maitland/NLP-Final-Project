from __future__ import annotations

from pathlib import Path

from .config import PathConfig, get_paths
from .schemas import Document
from .utils import (
    RagFaqError,
    infer_topic_from_identifier,
    make_source_id,
    read_csv_rows,
    read_text_with_metadata,
)

SUPPORTED_SUFFIXES = {".md", ".txt", ".html", ".htm", ".csv"}
FAQ_REQUIRED_COLUMNS = {"source_id", "question", "answer", "topic", "difficulty"}


def discover_knowledge_files(paths: PathConfig | None = None) -> list[Path]:
    paths = paths or get_paths()
    if not paths.knowledge_base_dir.exists():
        return []
    files = [
        path
        for path in sorted(paths.knowledge_base_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def _relative_source(path: Path, paths: PathConfig) -> str:
    return str(path.relative_to(paths.root_dir))


def _load_text_document(path: Path, paths: PathConfig) -> Document:
    text, frontmatter, title = read_text_with_metadata(path)
    source_id = make_source_id(path)
    topic = frontmatter.get("topic") or infer_topic_from_identifier(source_id)
    metadata = {
        "source": _relative_source(path, paths),
        "source_id": source_id,
        "topic": topic,
        "kind": frontmatter.get("kind", "doc"),
    }
    return Document(
        source_id=source_id,
        title=title,
        text=text,
        metadata=metadata,
    )


def _load_faq_documents(path: Path, paths: PathConfig) -> list[Document]:
    rows = read_csv_rows(path)
    if not rows:
        return []
    missing = FAQ_REQUIRED_COLUMNS.difference(rows[0].keys())
    if missing:
        raise RagFaqError(
            f"FAQ CSV {path.name} is missing required columns: {', '.join(sorted(missing))}"
        )

    documents: list[Document] = []
    seen_row_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        source_id = row["source_id"].strip()
        question = row["question"].strip()
        answer = row["answer"].strip()
        topic = row["topic"].strip()
        difficulty = row["difficulty"].strip()
        if not all([source_id, question, answer, topic, difficulty]):
            raise RagFaqError(
                f"FAQ CSV {path.name} has an empty required value on row {index + 1}."
            )
        if source_id in seen_row_ids:
            raise RagFaqError(f"FAQ CSV {path.name} repeats source_id {source_id!r}.")
        seen_row_ids.add(source_id)
        documents.append(
            Document(
                source_id=source_id,
                title=question,
                text=f"Question: {question}\nAnswer: {answer}",
                metadata={
                    "source": _relative_source(path, paths),
                    "source_id": source_id,
                    "topic": infer_topic_from_identifier(topic),
                    "difficulty": difficulty,
                    "kind": "faq",
                },
            )
        )
    return documents


def load_documents(paths: PathConfig | None = None) -> list[Document]:
    paths = paths or get_paths()
    documents: list[Document] = []
    seen_source_ids: set[str] = set()
    for path in discover_knowledge_files(paths):
        if path.suffix.lower() == ".csv":
            next_documents = _load_faq_documents(path, paths)
        else:
            next_documents = [_load_text_document(path, paths)]
        for document in next_documents:
            if document.source_id in seen_source_ids:
                raise RagFaqError(f"Duplicate source_id detected in knowledge base: {document.source_id}")
            seen_source_ids.add(document.source_id)
            documents.append(document)
    return documents


def inspect_documents(paths: PathConfig | None = None) -> dict[str, object]:
    documents = load_documents(paths)
    files = discover_knowledge_files(paths)
    return {
        "source_file_count": len(files),
        "faq_row_count": sum(1 for document in documents if document.metadata.get("kind") == "faq"),
        "source_ids": [document.source_id for document in documents],
        "titles": [document.title for document in documents],
        "topics": sorted({document.metadata.get("topic", "general") for document in documents}),
    }
