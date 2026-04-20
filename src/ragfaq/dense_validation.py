from __future__ import annotations

from datetime import datetime, timezone

from . import embeddings as embedding_module
from . import vector_store as vector_store_module
from .chunking import chunk_documents
from .config import (
    DENSE_MODEL_NAME,
    DENSE_TOP_K,
    DENSE_VALIDATION_COLLECTION_NAME,
    DENSE_VALIDATION_QUESTION,
    PathConfig,
    ensure_runtime_directories,
    get_paths,
)
from .ingest import load_documents
from .utils import RagFaqError, dump_json, load_json

DENSE_VALIDATION_START = "<!-- dense-validation:start -->"
DENSE_VALIDATION_END = "<!-- dense-validation:end -->"
DENSE_VALIDATION_SUMMARY_RELATIVE_PATH = "results/dense_validation_summary.json"
DENSE_VALIDATION_REPORT_RELATIVE_PATH = "results/dense_validation_report.md"

_CHECK_SEQUENCE = [
    ("chromadb_import", "Probe chromadb import"),
    ("sentence_transformers_import", "Probe sentence_transformers import"),
    ("local_model_cache", "Verify local MiniLM cache presence"),
    ("local_model_load", "Load MiniLM with local_files_only=True"),
    ("load_knowledge_base", "Load repo documents and chunks"),
    ("build_chroma_dir", "Prepare Chroma validation collection"),
    ("collection_add", "Store chunks with collection.add(...)"),
    ("stored_count", "Verify stored chunk count"),
    ("collection_query", "Query Chroma with n_results=3"),
    ("top_3_results", "Verify exactly top-3 chunks are returned"),
    ("source_references", "Verify source references are returned"),
]


def _empty_checks() -> list[dict[str, object]]:
    return [
        {
            "id": check_id,
            "name": check_name,
            "status": "not_run",
            "detail": "",
        }
        for check_id, check_name in _CHECK_SEQUENCE
    ]


def _set_check(
    checks: list[dict[str, object]],
    check_id: str,
    status: str,
    detail: str,
    **extra: object,
) -> None:
    for check in checks:
        if check["id"] != check_id:
            continue
        check["status"] = status
        check["detail"] = detail
        for key, value in extra.items():
            check[key] = value
        return
    raise AssertionError(f"Unknown dense validation check: {check_id}")


def _finish_summary(
    *,
    status: str,
    reason: str,
    checks: list[dict[str, object]],
    collection_name: str,
    question: str,
    chunk_count: int = 0,
    stored_count: int = 0,
    retrieved_chunks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    retrieved_chunks = retrieved_chunks or []
    return {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": DENSE_MODEL_NAME,
        "collection_name": collection_name,
        "question": question,
        "top_k": DENSE_TOP_K,
        "reason": reason,
        "chunk_count": chunk_count,
        "stored_count": stored_count,
        "retrieved_count": len(retrieved_chunks),
        "retrieved_chunks": retrieved_chunks,
        "checks": checks,
    }


def _retrieved_chunk_payload(chunk) -> dict[str, object]:
    return {
        "rank": chunk.rank,
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "source": chunk.metadata.get("source", ""),
        "topic": chunk.metadata.get("topic", "general"),
        "score": chunk.score,
        "distance": chunk.distance,
    }


def run_dense_validation(
    paths: PathConfig | None = None,
    *,
    collection_name: str = DENSE_VALIDATION_COLLECTION_NAME,
    question: str = DENSE_VALIDATION_QUESTION,
) -> dict[str, object]:
    paths = ensure_runtime_directories(paths or get_paths())
    checks = _empty_checks()

    try:
        vector_store_module._load_chromadb_module()
    except Exception as exc:
        reason = f"chromadb import unavailable: {type(exc).__name__}: {exc}"
        _set_check(checks, "chromadb_import", "skipped", reason)
        return _finish_summary(
            status="skipped",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
        )
    _set_check(checks, "chromadb_import", "passed", "chromadb imported successfully.")

    try:
        embedding_module._load_sentence_transformer_class()
    except Exception as exc:
        reason = (
            "sentence_transformers import unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        _set_check(checks, "sentence_transformers_import", "skipped", reason)
        return _finish_summary(
            status="skipped",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
        )
    _set_check(
        checks,
        "sentence_transformers_import",
        "passed",
        "sentence_transformers imported successfully.",
    )

    provider = embedding_module.SentenceTransformerEmbeddingProvider(paths.cache_dir)
    cached, cache_detail = provider.local_model_status()
    if not cached:
        reason = (
            "MiniLM local cache unavailable: "
            f"{cache_detail}"
        )
        _set_check(checks, "local_model_cache", "skipped", reason)
        return _finish_summary(
            status="skipped",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
        )
    _set_check(
        checks,
        "local_model_cache",
        "passed",
        f"MiniLM cache found at {cache_detail}.",
        cache_path=cache_detail,
    )

    try:
        provider.ensure_model_loaded()
    except RagFaqError as exc:
        reason = str(exc)
        _set_check(checks, "local_model_load", "skipped", reason)
        return _finish_summary(
            status="skipped",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
        )
    _set_check(
        checks,
        "local_model_load",
        "passed",
        "MiniLM loaded successfully with local_files_only=True.",
    )

    try:
        documents = load_documents(paths)
        chunks = chunk_documents(documents)
    except RagFaqError as exc:
        reason = str(exc)
        _set_check(checks, "load_knowledge_base", "failed", reason)
        return _finish_summary(
            status="failed",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
        )

    chunk_count = len(chunks)
    if chunk_count < DENSE_TOP_K:
        reason = (
            f"Dense validation requires at least {DENSE_TOP_K} chunks, "
            f"but only {chunk_count} were available."
        )
        _set_check(
            checks,
            "load_knowledge_base",
            "failed",
            reason,
            document_count=len(documents),
            chunk_count=chunk_count,
        )
        return _finish_summary(
            status="failed",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
            chunk_count=chunk_count,
        )
    _set_check(
        checks,
        "load_knowledge_base",
        "passed",
        f"Loaded {len(documents)} documents and {chunk_count} chunks.",
        document_count=len(documents),
        chunk_count=chunk_count,
    )

    store = vector_store_module.ChromaVectorStore(paths.chroma_dir, collection_name=collection_name)
    _set_check(
        checks,
        "build_chroma_dir",
        "passed",
        f"Validation collection {collection_name} will be rebuilt under {paths.chroma_dir}.",
        chroma_dir=str(paths.chroma_dir),
    )

    try:
        embeddings = provider.embed_texts([chunk.text for chunk in chunks])
        stored_count = store.index(chunks, embeddings, rebuild=True)
    except RagFaqError as exc:
        reason = str(exc)
        _set_check(checks, "collection_add", "failed", reason)
        return _finish_summary(
            status="failed",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
            chunk_count=chunk_count,
        )
    _set_check(
        checks,
        "collection_add",
        "passed",
        f"collection.add(...) stored {chunk_count} chunk documents.",
        stored_count=stored_count,
    )

    if stored_count != chunk_count:
        reason = (
            "Stored chunk count mismatch after collection.add(...): "
            f"expected {chunk_count}, observed {stored_count}."
        )
        _set_check(
            checks,
            "stored_count",
            "failed",
            reason,
            expected_count=chunk_count,
            observed_count=stored_count,
        )
        return _finish_summary(
            status="failed",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
            chunk_count=chunk_count,
            stored_count=stored_count,
        )
    _set_check(
        checks,
        "stored_count",
        "passed",
        f"Stored chunk count matches the expected {chunk_count} chunks.",
        expected_count=chunk_count,
        observed_count=stored_count,
    )

    try:
        query_embedding = provider.embed_texts([question])[0]
        retrieved = store.query(query_embedding, top_k=DENSE_TOP_K)
    except (IndexError, RagFaqError) as exc:
        reason = str(exc)
        _set_check(checks, "collection_query", "failed", reason)
        return _finish_summary(
            status="failed",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
            chunk_count=chunk_count,
            stored_count=stored_count,
        )
    _set_check(
        checks,
        "collection_query",
        "passed",
        f"collection.query(..., n_results={DENSE_TOP_K}) executed successfully.",
    )

    if len(retrieved) != DENSE_TOP_K:
        reason = (
            f"Expected exactly {DENSE_TOP_K} retrieved chunks, "
            f"but received {len(retrieved)}."
        )
        _set_check(
            checks,
            "top_3_results",
            "failed",
            reason,
            expected_count=DENSE_TOP_K,
            observed_count=len(retrieved),
        )
        return _finish_summary(
            status="failed",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
            chunk_count=chunk_count,
            stored_count=stored_count,
            retrieved_chunks=[_retrieved_chunk_payload(chunk) for chunk in retrieved],
        )
    _set_check(
        checks,
        "top_3_results",
        "passed",
        f"Retrieved exactly {DENSE_TOP_K} chunks from Chroma.",
        observed_count=len(retrieved),
    )

    retrieved_payload = [_retrieved_chunk_payload(chunk) for chunk in retrieved]
    missing_reference_ranks = [
        chunk.rank
        for chunk in retrieved
        if not (chunk.chunk_id and chunk.source_id and chunk.metadata.get("source"))
    ]
    if missing_reference_ranks:
        reason = (
            "Missing source references in retrieved chunks at ranks "
            f"{', '.join(str(rank) for rank in missing_reference_ranks)}."
        )
        _set_check(
            checks,
            "source_references",
            "failed",
            reason,
            missing_ranks=missing_reference_ranks,
        )
        return _finish_summary(
            status="failed",
            reason=reason,
            checks=checks,
            collection_name=collection_name,
            question=question,
            chunk_count=chunk_count,
            stored_count=stored_count,
            retrieved_chunks=retrieved_payload,
        )
    _set_check(
        checks,
        "source_references",
        "passed",
        f"All {DENSE_TOP_K} retrieved chunks include chunk_id, source_id, and source metadata.",
    )

    return _finish_summary(
        status="passed",
        reason="",
        checks=checks,
        collection_name=collection_name,
        question=question,
        chunk_count=chunk_count,
        stored_count=stored_count,
        retrieved_chunks=retrieved_payload,
    )


def _escape_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def render_dense_validation_report(summary_payload: dict[str, object] | None) -> str:
    if not summary_payload:
        return (
            "# Dense Validation Report\n\n"
            "_Dense validation has not been run yet. "
            "Run `python scripts/validate_dense_path.py` to generate real artifacts._\n"
        )

    status = str(summary_payload.get("status", "unknown"))
    reason = str(summary_payload.get("reason", "") or "").replace("\n", " ")
    checks = list(summary_payload.get("checks", []))
    retrieved_chunks = list(summary_payload.get("retrieved_chunks", []))

    lines = [
        "# Dense Validation Report",
        "",
        "## Summary",
        "",
        f"- Status: `{status}`",
        f"- Generated at: `{summary_payload.get('generated_at', 'unknown')}`",
        f"- Model: `{summary_payload.get('model_name', DENSE_MODEL_NAME)}`",
        f"- Collection: `{summary_payload.get('collection_name', DENSE_VALIDATION_COLLECTION_NAME)}`",
        f"- Question: `{summary_payload.get('question', DENSE_VALIDATION_QUESTION)}`",
        f"- Expected top-k: `{summary_payload.get('top_k', DENSE_TOP_K)}`",
    ]
    if reason:
        lines.append(f"- Reason: {reason}")
    if status == "passed":
        lines.append(
            "- Outcome: The course-required dense retrieval path validated successfully."
        )
    else:
        lines.append(
            "- Outcome: The dense path did not validate successfully in this environment."
        )

    lines.extend(
        [
            "",
            "## Check Results",
            "",
            "| Step | Status | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for check in checks:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(check.get("name", check.get("id", "unknown"))),
                    _escape_cell(check.get("status", "unknown")),
                    _escape_cell(check.get("detail", "")),
                ]
            )
            + " |"
        )

    if retrieved_chunks:
        lines.extend(["", "## Retrieved Top-3 Chunks", ""])
        for chunk in retrieved_chunks:
            lines.append(
                "- "
                + ", ".join(
                    [
                        f"rank={chunk.get('rank', '?')}",
                        f"chunk_id={chunk.get('chunk_id', 'unknown')}",
                        f"source_id={chunk.get('source_id', 'unknown')}",
                        f"source={chunk.get('source', 'unknown')}",
                    ]
                )
            )

    return "\n".join(lines).rstrip() + "\n"


def render_dense_validation_section(summary_payload: dict[str, object] | None) -> str:
    lines = [
        "Auto-generated from real dense validation artifacts under "
        f"`{DENSE_VALIDATION_SUMMARY_RELATIVE_PATH}` and `{DENSE_VALIDATION_REPORT_RELATIVE_PATH}`.",
        "",
        "### Implemented Dense Path",
        "",
        "- Dense retrieval is implemented with `sentence-transformers/all-MiniLM-L6-v2` embeddings and a ChromaDB collection.",
        "- The code path stores chunks through `collection.add(...)` and queries them through `collection.query(..., n_results=3)`.",
        "",
        "### Validated Dense Path",
        "",
    ]

    if not summary_payload:
        lines.extend(
            [
                "- Status: `not_run`",
                "- No dense validation artifacts have been generated yet.",
                "- Run `python scripts/validate_dense_path.py` to record whether the dense path passes or is skipped in the current environment.",
            ]
        )
    else:
        status = str(summary_payload.get("status", "unknown"))
        reason = str(summary_payload.get("reason", "") or "").replace("\n", " ")
        lines.append(f"- Latest status: `{status}`")
        lines.append(f"- Latest artifact timestamp: `{summary_payload.get('generated_at', 'unknown')}`")
        if status == "passed":
            lines.append("- The dense path was validated successfully in this environment.")
            lines.append(
                "- The validation run confirmed that `collection.add(...)` stored the chunk corpus and `collection.query(..., n_results=3)` returned exactly three chunks with source references."
            )
        elif status == "skipped":
            lines.append("- The dense path is implemented, but it was not validated successfully in this environment.")
        else:
            lines.append("- The dense path was exercised, but the validation run did not succeed.")
        if reason:
            lines.append(f"- Reason: {reason}")
        lines.append(
            f"- Detailed evidence: `{DENSE_VALIDATION_SUMMARY_RELATIVE_PATH}` and `{DENSE_VALIDATION_REPORT_RELATIVE_PATH}`."
        )

    lines.extend(
        [
            "",
            "### Offline Fallback Path",
            "",
            "- The offline-safe default remains TF-IDF retrieval with `--llm offline`.",
            "- In `--backend auto`, the CLI falls back to TF-IDF with a friendly explanation whenever the dense stack or local MiniLM runtime is unavailable.",
        ]
    )
    return "\n".join(lines)


def _replace_marked_section(text: str, replacement: str) -> str:
    start_index = text.find(DENSE_VALIDATION_START)
    end_index = text.find(DENSE_VALIDATION_END)
    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise RagFaqError("dense validation markers were not found")
    end_index += len(DENSE_VALIDATION_END)
    return (
        text[:start_index]
        + DENSE_VALIDATION_START
        + "\n"
        + replacement.strip()
        + "\n"
        + DENSE_VALIDATION_END
        + text[end_index:]
    )


def sync_dense_validation_project_report(
    paths: PathConfig | None = None,
    summary_payload: dict[str, object] | None = None,
) -> None:
    paths = paths or get_paths()
    if summary_payload is None:
        loaded = load_json(paths.dense_validation_summary_path, default=None)
        summary_payload = loaded if isinstance(loaded, dict) and loaded else None
    replacement = render_dense_validation_section(summary_payload)
    original = paths.project_report_path.read_text(encoding="utf-8")
    updated = _replace_marked_section(original, replacement)
    paths.project_report_path.write_text(updated, encoding="utf-8")


def write_dense_validation_artifacts(
    summary_payload: dict[str, object],
    paths: PathConfig | None = None,
) -> None:
    paths = paths or get_paths()
    dump_json(paths.dense_validation_summary_path, summary_payload)
    paths.dense_validation_report_path.write_text(
        render_dense_validation_report(summary_payload),
        encoding="utf-8",
    )
    sync_dense_validation_project_report(paths, summary_payload=summary_payload)
