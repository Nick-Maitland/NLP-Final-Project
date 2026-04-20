from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.config import get_paths, get_runtime_availability
from ragfaq.ingest import inspect_documents
from ragfaq.retrievers import inspect_index_state


def main() -> int:
    paths = get_paths()
    availability = get_runtime_availability()
    docs = inspect_documents(paths)
    indexes = inspect_index_state(paths)

    print(f"Root: {paths.root_dir}")
    print(f"Knowledge-base documents: {docs['source_file_count']}")
    print(f"Source IDs: {', '.join(docs['source_ids'])}")
    print(f"Lexical index ready: {indexes['lexical_index_ready']}")
    print(f"Dense index ready: {indexes['dense_index_ready']}")
    print(f"Chroma SDK available: {availability.chromadb.available}")
    print(f"Sentence-transformers available: {availability.sentence_transformers.available}")
    print(f"OpenAI SDK available: {availability.openai_sdk.available}")
    print(f"OPENAI_API_KEY present: {availability.openai_key_available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

