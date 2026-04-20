from __future__ import annotations

from pathlib import Path

from .config import PathConfig, get_paths
from .schemas import Document
from .utils import extract_title, make_source_id, read_supported_text

SUPPORTED_SUFFIXES = {".md", ".txt", ".html", ".htm"}


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


def load_documents(paths: PathConfig | None = None) -> list[Document]:
    paths = paths or get_paths()
    documents: list[Document] = []
    for path in discover_knowledge_files(paths):
        text = read_supported_text(path)
        title = extract_title(path, text)
        source_id = make_source_id(path)
        documents.append(
            Document(
                source_id=source_id,
                title=title,
                text=text,
                metadata={"path": str(path.relative_to(paths.root_dir))},
            )
        )
    return documents


def inspect_documents(paths: PathConfig | None = None) -> dict[str, object]:
    documents = load_documents(paths)
    return {
        "source_file_count": len(documents),
        "source_ids": [document.source_id for document in documents],
        "titles": [document.title for document in documents],
    }

