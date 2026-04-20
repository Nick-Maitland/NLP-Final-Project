from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.chunking import chunk_documents, chunk_documents_with_report
from ragfaq.config import (
    COLLECTION_NAME,
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
    DENSE_TOP_K,
    ensure_runtime_directories,
    get_paths,
)
from ragfaq.demo import build_demo_trace_payload, demo_questions_for_run, run_demo, write_demo_markdown
from ragfaq.evaluation import run_evaluation
from ragfaq.generation import answer_question
from ragfaq.ingest import discover_knowledge_files, load_documents
from ragfaq.retrievers import inspect_index_state, maybe_build_indexes, retrieve
from ragfaq.schemas import BackendMode, LlmMode
from ragfaq.utils import RagFaqError, dump_json, format_sources, normalize_text


COMPATIBILITY_TEXT = (
    "Compatibility aliases remain supported: "
    "--build-index, --ask \"QUESTION\", --evaluate, and --smoke-test --offline."
)
DEFAULT_TRACE_ARG = "__DEFAULT_TRACE_OUTPUT__"


def _choices(enum_cls) -> list[str]:
    return [item.value for item in enum_cls]


def _legacy_to_subcommand(argv: list[str]) -> list[str]:
    args = list(argv)
    if not args:
        return args
    if args[0] in {"build", "ask", "evaluate", "inspect-kb", "demo"}:
        return args

    if "--build-index" in args:
        args.remove("--build-index")
        return ["build", *args]

    if "--ask" in args:
        index = args.index("--ask")
        try:
            question = args[index + 1]
        except IndexError as exc:
            raise RagFaqError("Legacy --ask mode requires a question string.") from exc
        remaining = args[:index] + args[index + 2 :]
        return ["ask", "--question", question, *remaining]

    if "--evaluate" in args:
        args.remove("--evaluate")
        return ["evaluate", *args]

    if "--smoke-test" in args:
        args.remove("--smoke-test")
        rewritten = ["demo", *args]
        if "--backend" not in rewritten:
            rewritten.extend(["--backend", BackendMode.TFIDF.value])
        if "--llm" not in rewritten and "--offline" not in rewritten:
            rewritten.extend(["--llm", LlmMode.OFFLINE.value])
        if "--question" not in rewritten:
            rewritten.extend(["--question", "What is self-attention?"])
        return rewritten

    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Course Project 10 RAG FAQ system.",
        epilog=COMPATIBILITY_TEXT,
    )
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--backend",
        choices=_choices(BackendMode),
        default=BackendMode.AUTO.value,
        help="Retrieval backend to use.",
    )
    common.add_argument(
        "--llm",
        choices=_choices(LlmMode),
        default=LlmMode.AUTO.value,
        help="LLM mode to use.",
    )
    common.add_argument(
        "--offline",
        action="store_true",
        help="Convenience alias for --llm offline.",
    )
    common.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )
    common.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help="Chroma collection name for dense retrieval.",
    )
    retrieval_common = argparse.ArgumentParser(add_help=False)
    retrieval_common.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Candidate pool size for hybrid retrieval before fusion and MMR.",
    )
    retrieval_common.add_argument(
        "--show-context",
        action="store_true",
        help="Print the full text of the final retrieved chunks.",
    )
    retrieval_common.add_argument(
        "--trace-output",
        nargs="?",
        const=DEFAULT_TRACE_ARG,
        default=None,
        help="Write a retrieval trace JSON. Omit a value to use results/traces/latest_retrieval_trace.json.",
    )

    build_parser = subparsers.add_parser(
        "build",
        parents=[common],
        help="Build indexes from the knowledge base.",
    )
    build_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear and rebuild the target Chroma collection before indexing.",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        parents=[common, retrieval_common],
        help="Ask a question against the indexed knowledge base.",
    )
    ask_parser.add_argument("--question", required=True, help="Question to answer.")

    subparsers.add_parser(
        "evaluate",
        parents=[common, retrieval_common],
        help="Run the 30-question evaluation and refresh the report.",
    )
    subparsers.add_parser(
        "inspect-kb",
        parents=[common],
        help="Inspect repo-local knowledge-base files and index availability.",
    )
    demo_parser = subparsers.add_parser(
        "demo",
        parents=[common, retrieval_common],
        help="Run an offline-safe smoke demo.",
    )
    demo_parser.add_argument(
        "--question",
        default=None,
        help="Optional single demo question. Omit to run the curated five-question showcase.",
    )

    return parser


def _resolve_requested_modes(args: argparse.Namespace) -> tuple[BackendMode, LlmMode]:
    backend = BackendMode(args.backend)
    llm = LlmMode.OFFLINE if args.offline else LlmMode(args.llm)
    return backend, llm


def _resolve_trace_output_path(args: argparse.Namespace, paths) -> Path | None:
    trace_output = getattr(args, "trace_output", None)
    if trace_output is None:
        return None
    if trace_output == DEFAULT_TRACE_ARG:
        return paths.latest_trace_path
    candidate = Path(trace_output).expanduser()
    if not candidate.is_absolute():
        candidate = paths.root_dir / candidate
    return candidate


def _effective_top_k(args: argparse.Namespace, requested_backend: BackendMode) -> int:
    top_k = args.top_k
    if requested_backend is BackendMode.CHROMA and top_k != DENSE_TOP_K:
        print(
            "Chroma course mode note: top-k is fixed to 3 for direct Project 10 Chroma retrieval."
        )
        return DENSE_TOP_K
    return top_k


def _effective_candidate_k(args: argparse.Namespace, requested_backend: BackendMode, top_k: int) -> int:
    candidate_k = getattr(args, "candidate_k", DEFAULT_CANDIDATE_K)
    if requested_backend is BackendMode.HYBRID and candidate_k < top_k:
        raise RagFaqError("--candidate-k must be greater than or equal to --top-k for hybrid mode.")
    return max(candidate_k, top_k)


def _load_docs_and_chunks():
    paths = ensure_runtime_directories(get_paths())
    documents = load_documents(paths)
    if not documents:
        raise RagFaqError(
            "No knowledge-base files were found. Populate the root knowledge_base/ folder first."
        )
    chunks = chunk_documents(documents)
    return paths, documents, chunks


def _print_auto_backend_note(
    requested_backend: BackendMode,
    resolved_backend: BackendMode,
    paths,
    collection_name: str,
) -> None:
    if requested_backend is not BackendMode.AUTO or resolved_backend is not BackendMode.TFIDF:
        return
    index_state = inspect_index_state(paths, collection_name=collection_name)
    if index_state["dense_runtime_available"]:
        detail = "dense index is not built yet"
    else:
        detail = index_state["dense_runtime_reason"]
    print(
        "Auto backend fallback: using tfidf because dense retrieval is unavailable "
        f"({detail})."
    )


def _chunk_metric(chunk) -> str:
    parts = [f"score={chunk.score:.4f}"]
    if chunk.distance is not None:
        parts.insert(0, f"distance={chunk.distance:.4f}")
    if chunk.fusion_score is not None:
        parts.append(f"fusion={chunk.fusion_score:.4f}")
    if chunk.lexical_rank is not None:
        parts.append(f"lexical_rank={chunk.lexical_rank}")
    if chunk.dense_rank is not None:
        parts.append(f"dense_rank={chunk.dense_rank}")
    if chunk.mmr_score is not None:
        parts.append(f"mmr={chunk.mmr_score:.4f}")
    return " ".join(parts)


def _snippet(text: str, limit: int = 140) -> str:
    normalized = normalize_text(text).replace("\n", " ")
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _print_retrieval_preview(chunks) -> None:
    print("Retrieval results:")
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        print(
            f"{chunk.rank}. source={source} source_id={chunk.source_id} "
            f"{_chunk_metric(chunk)}"
        )
        print(f"   snippet={_snippet(chunk.text)}")
        if chunk.selection_reason:
            print(f"   why={chunk.selection_reason}")


def _print_context(chunks) -> None:
    print("CONTEXT")
    for chunk in chunks:
        print(f"[{chunk.rank}] {_snippet(chunk.text, limit=220)}")


def _print_sources(chunks) -> None:
    print("SOURCES")
    for chunk in chunks:
        topic = chunk.metadata.get("topic", "general")
        source = chunk.metadata.get("source", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", "0")
        print(f"[{chunk.rank}] {chunk.source_id} | {topic} | {source} | {chunk_index}")


def _print_answer_sections(answer, show_context: bool) -> None:
    answer_text = answer.answer_text or answer.answer
    print("ANSWER")
    print(answer_text)
    print("")
    _print_sources(answer.retrieved_chunks)
    for warning in answer.citation_warnings:
        print(f"WARNING: {warning}")
    if show_context:
        print("")
        _print_context(answer.retrieved_chunks)


def _write_trace(trace_payload: dict[str, object] | None, output_path: Path | None) -> None:
    if trace_payload is None or output_path is None:
        return
    dump_json(output_path, trace_payload)
    print(f"Trace output: {output_path}")


def command_build(args: argparse.Namespace) -> int:
    requested_backend, _ = _resolve_requested_modes(args)
    paths, documents, chunks = _load_docs_and_chunks()
    summary = maybe_build_indexes(
        chunks,
        requested_backend=requested_backend,
        paths=paths,
        collection_name=args.collection_name,
        rebuild=args.rebuild,
    )
    print(f"Knowledge-base documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    lexical_summary = summary["lexical_index"]
    print(f"Lexical index: ready at {lexical_summary['path']}")
    dense_summary = summary["dense_index"]
    if dense_summary.get("built"):
        print(
            f"Dense index: built with {dense_summary['document_count']} stored chunks "
            f"in collection {args.collection_name}"
        )
    else:
        print(f"Dense index: skipped ({dense_summary.get('reason', 'not requested')})")
        if requested_backend is BackendMode.AUTO:
            print(
                "Auto backend fallback: tfidf will remain the safe local default until "
                "the dense stack and MiniLM model are available."
            )
    return 0


def command_inspect_kb(args: argparse.Namespace) -> int:
    paths = ensure_runtime_directories(get_paths())
    knowledge_files = discover_knowledge_files(paths)
    documents = load_documents(paths)
    chunks, chunk_report = chunk_documents_with_report(documents)
    index_state = inspect_index_state(paths, collection_name=args.collection_name)
    print(f"Knowledge-base directory: {paths.knowledge_base_dir}")
    print(f"Chroma collection: {args.collection_name}")
    print(f"Knowledge-base files: {len(knowledge_files)}")
    print(
        "FAQ rows: "
        f"{sum(1 for document in documents if document.metadata.get('kind') == 'faq')}"
    )
    print(f"Source documents: {len(documents)}")
    print(f"Chunk count (computed): {len(chunks)}")
    print(f"Topics covered: {format_sources(chunk_report['topics'])}")
    print(f"Top 10 chunk IDs: {format_sources(chunk_report['top_chunk_ids'])}")
    if chunk_report["too_short_chunk_ids"]:
        print(
            "Too-short chunks: "
            f"{format_sources(chunk_report['too_short_chunk_ids'])}"
        )
    else:
        print("Too-short chunks: none")
    if chunk_report["duplicate_chunk_ids"]:
        print(
            "Duplicate-like chunks skipped: "
            f"{len(chunk_report['duplicate_chunk_ids'])} "
            f"({format_sources(chunk_report['duplicate_chunk_ids'][:10])})"
        )
    else:
        print("Duplicate-like chunks skipped: 0")
    print(f"Lexical index ready: {index_state['lexical_index_ready']}")
    print(f"Dense index ready: {index_state['dense_index_ready']}")
    print(f"Chroma SDK available: {index_state['chroma_sdk_available']}")
    print(
        f"Sentence-transformers available: {index_state['sentence_transformers_available']}"
    )
    if not index_state["dense_runtime_available"]:
        print(f"Dense availability reason: {index_state['dense_runtime_reason']}")
    print(f"OpenAI SDK available: {index_state['openai_sdk_available']}")
    print(f"OPENAI_API_KEY present: {index_state['openai_key_available']}")
    return 0


def command_ask(args: argparse.Namespace) -> int:
    requested_backend, requested_llm = _resolve_requested_modes(args)
    paths = ensure_runtime_directories(get_paths())
    top_k = _effective_top_k(args, requested_backend)
    candidate_k = _effective_candidate_k(args, requested_backend, top_k)
    retrieval = retrieve(
        question=args.question,
        requested_backend=requested_backend,
        top_k=top_k,
        candidate_k=candidate_k,
        paths=paths,
        collection_name=args.collection_name,
    )
    _print_auto_backend_note(
        requested_backend,
        retrieval.resolved_backend,
        paths,
        args.collection_name,
    )
    answer = answer_question(
        question=args.question,
        retrieved_chunks=retrieval.chunks,
        requested_llm=requested_llm,
        resolved_backend=retrieval.resolved_backend,
    )
    print(f"Question: {args.question}")
    print(f"Resolved backend: {answer.resolved_backend.value}")
    print(f"Resolved llm: {answer.resolved_llm.value}")
    _print_retrieval_preview(answer.retrieved_chunks)
    print("")
    _print_answer_sections(answer, args.show_context)
    _write_trace(retrieval.trace, _resolve_trace_output_path(args, paths))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    requested_backend, requested_llm = _resolve_requested_modes(args)
    paths, _, chunks = _load_docs_and_chunks()
    top_k = _effective_top_k(args, requested_backend)
    candidate_k = _effective_candidate_k(args, requested_backend, top_k)
    build_summary = maybe_build_indexes(
        chunks,
        requested_backend=requested_backend,
        paths=paths,
        collection_name=args.collection_name,
    )
    if requested_backend is BackendMode.AUTO and not build_summary["dense_index"].get("built"):
        print(
            "Auto backend fallback: evaluation is using tfidf because dense retrieval "
            f"is unavailable ({build_summary['dense_index'].get('reason', 'unknown reason')})."
        )
    results, summary = run_evaluation(
        requested_backend=requested_backend,
        requested_llm=requested_llm,
        paths=paths,
        top_k=top_k,
        candidate_k=candidate_k,
        collection_name=args.collection_name,
        show_context=args.show_context,
        trace_output_path=_resolve_trace_output_path(args, paths),
    )
    print(f"Evaluated rows: {len(results)}")
    print(f"Answerable rows: {summary['answerable_count']}")
    print(f"Unanswerable rows: {summary['unanswerable_count']}")
    print(
        "Retrieval Recall@3 (answerable): "
        f"{summary['retrieval_recall_at_3_answerable']:.2f}"
    )
    print(f"MRR@3 (answerable): {summary['mrr_at_3_answerable']:.2f}")
    print(f"Average faithfulness: {summary['faithfulness_avg']:.2f}")
    print(f"Citation valid rate: {summary['citation_valid_rate']:.2f}")
    abstention_accuracy = summary["abstention_accuracy_unanswerable"]
    if abstention_accuracy is not None:
        print(f"Abstention accuracy (unanswerable): {abstention_accuracy:.2f}")
    print(f"Average latency (ms): {summary['avg_latency_ms']:.2f}")
    print(f"Updated CSV: {paths.test_questions_path}")
    print(f"Scored CSV: {paths.scored_questions_path}")
    print(f"Summary JSON: {paths.evaluation_summary_path}")
    print(f"Evaluation report: {paths.evaluation_report_path}")
    print(f"Updated report: {paths.failure_report_path}")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    requested_backend, requested_llm = _resolve_requested_modes(args)
    paths, _, chunks = _load_docs_and_chunks()
    top_k = _effective_top_k(args, requested_backend)
    candidate_k = _effective_candidate_k(args, requested_backend, top_k)
    if not paths.lexical_index_path.exists():
        maybe_build_indexes(
            chunks,
            requested_backend=BackendMode.TFIDF,
            paths=paths,
            collection_name=args.collection_name,
        )
    questions = demo_questions_for_run(args.question)
    entries = run_demo(
        requested_backend=requested_backend,
        requested_llm=requested_llm,
        questions=questions,
        paths=paths,
        top_k=top_k,
        candidate_k=candidate_k,
        collection_name=args.collection_name,
    )
    print("Demo mode")
    print(f"Questions: {len(entries)}")
    if any(entry.answer.resolved_backend is BackendMode.TFIDF for entry in entries):
        _print_auto_backend_note(
            requested_backend,
            BackendMode.TFIDF,
            paths,
            args.collection_name,
        )
    for index, entry in enumerate(entries, start=1):
        print("")
        print(f"Demo Question {index}/{len(entries)}")
        print(f"Question: {entry.question}")
        print(f"Resolved backend: {entry.answer.resolved_backend.value}")
        print(f"Resolved llm: {entry.answer.resolved_llm.value}")
        print(f"Latency (ms): {entry.latency_ms:.2f}")
        print(f"Offline fallback used: {str(entry.offline_fallback_used).lower()}")
        _print_retrieval_preview(entry.answer.retrieved_chunks)
        print("")
        _print_answer_sections(entry.answer, args.show_context)
    write_demo_markdown(
        entries,
        requested_backend=requested_backend,
        requested_llm=requested_llm,
        paths=paths,
        show_context=args.show_context,
    )
    print("")
    print(f"Demo markdown: {paths.demo_run_path}")
    _write_trace(
        build_demo_trace_payload(entries),
        _resolve_trace_output_path(args, paths),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    rewritten = _legacy_to_subcommand(argv)
    args = parser.parse_args(rewritten)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "build": command_build,
        "ask": command_ask,
        "evaluate": command_evaluate,
        "inspect-kb": command_inspect_kb,
        "demo": command_demo,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RagFaqError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
